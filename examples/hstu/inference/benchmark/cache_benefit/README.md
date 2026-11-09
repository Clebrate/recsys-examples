# KV / embedding cache benefit benchmark

用于合作实验的测量工具：复用同目录上级 `cache_ratio_sweep.py` 的请求构造、模型构建和 HSTU KV-window forward，测量 **hit 相比 miss 节省多少完整请求延迟**，以及 cache ratio 改变时两侧命中率和 E2E 如何变化。

代码是实验版固定页 GPU cache 容器，正常替换采用 LRU，resize 时回收物理页并驱逐受影响条目。它不是 NVE／FlexKV 的在线 resize API，也不包含已经验证胜出的在线分配算法。

默认沿用场景 3 的双窗口语义：embedding 查询完整的 history cap 加候选，HSTU／KV 只使用最近 `--kv-history-items=128` 个历史 item 及其 action。KV 收益与所需容量均针对这个范围；可在逐请求记录中核对 embedding lookup 数和 KV history token 数。

## 测什么

| 模式 | 操作 | 主要结果 |
| --- | --- | --- |
| `paired` | 对同一请求重复测 KV／embedding 的 HH、MH、HM、MM 四种状态，随机化测量顺序 | KV 与 embedding 的条件收益、交互项、各状态 mean／p50／p95／p99 |
| `replay` | 固定预算和比例连续重放；可串行比较多个固定比例 | 请求级 KV hit、token hit、embedding key hit、embedding 请求全 hit、miss IDs、E2E、分窗统计 |
| `all-hit` | GPU 容量覆盖真实 working set，预填充所有用户，再验证每条请求实际全中 | 全命中 E2E、所需 KV／embedding 容量、实际回源次数为零 |

`H/M` 的第一个字符代表 KV，第二个代表 embedding。Embedding M 表示计时前该请求所有 **unique keys** 都已从 GPU cache 失效；重复 key 在同一请求内首次回填后可以命中，因此这一状态的 occurrence/key hit rate 不一定为 0。

收益定义为 `L_miss − L_hit`，单位 **ms/request**。分别报告：

```text
KV saving | embedding hit  = L(MH) − L(HH)
KV saving | embedding miss = L(MM) − L(HM)
EMB saving | KV hit        = L(HM) − L(HH)
EMB saving | KV miss       = L(MM) − L(MH)
interaction                = L(MM) − L(HM) − L(MH) + L(HH)
```

四种状态均检查实际 miss 数、KV 覆盖长度及输出一致性，无法实现目标状态时拒绝这次测量。负收益如实保留。Bootstrap 区间按请求聚类，重复测量不当成新的独立请求；区间反映本次样本差异，跨运行／硬件波动需另做重复实验。

`paired` 默认从请求池均匀、无放回选择用户请求，结果是该样本的平均收益。它不自动成为 Zipf 或其他业务流量的请求加权收益；按业务流量估算总收益时还需匹配请求分布、miss 数与命中条件。

`paired` 的预填充／失效操作是计时外的对照准备，用于隔离收益。`replay` 保持真实缓存历史；其不同 hit/miss 请求的延迟分组是描述统计，不能直接当作同请求的因果收益。完整 online 方案仍需在连续重放中计入控制、驱逐、回填与恢复成本。

## 环境与数据

使用本仓库已配置好的 HSTU inference 环境：CUDA GPU、PyTorch、TorchRec、gin、NumPy、HSTU CUDA 扩展、`recsys_kvcache_manager` 与匹配的 NVE Python 包。NVE 仅用于沿用原模型构造流程；实验 cache 在首次推理前替换进去。NVE 版本选择遵循 `modules/nve_compat.py`，按环境设置 `PYTHONPATH`／`NVE_INSTALL_ROOT`／`NVE_VERSION`。

以下命令从仓库的 `examples/hstu` 执行，建议使用配置好的容器和 `python3 -s`。工具通过 `HSTU_INFERENCE_ONLY=1` 选择推理路径。

没有提供数据时，使用可复现的 synthetic 请求。实际 CSV 采用原 sweep 格式：

```csv
user_id,video_id,action_weights
```

后两列是等长 JSON ID 数组。每个 user key 对应一条不变的请求；历史改变时需使用版本化 cache key。也可通过 `--pool` 读取已有 `DYNPOOL1` 目录。数据和运行日志不随代码发布。

默认不加载 checkpoint，video 参数采用原 sweep 的 default-zero 语义；action 参数固定种子初始化。`--parameters pattern` 使用确定的非零测试值检查数据正确性。这是缓存性能／机制 benchmark，不评价推荐准确率或真实生产参数服务器性能。

## 快速运行

同请求四状态测量，不添加模拟等待：

```bash
python3 -s inference/benchmark/cache_benefit/run.py paired \
  --history-cap 160 --synthetic-users 32 --parameters pattern \
  --budget-gib 0.25 --fraction 0.55 --samples 16 --repeats 3 \
  --output /tmp/hstu-paired
```

场景 3 原始请求池上的受控成本模型：

```bash
python3 -s inference/benchmark/cache_benefit/run.py paired \
  --data /path/to/processed_seqs.csv --history-cap 4096 --candidates 100 \
  --budget-gib 16 --fraction 0.55 --samples 256 --repeats 3 \
  --kv-miss-ms 1.329810 --embedding-us-per-id 0.410387 \
  --output /tmp/hstu-paired-modeled
```

模拟一次完整 MGET 为 7.5 ms：

```bash
python3 -s inference/benchmark/cache_benefit/run.py paired \
  --history-cap 160 --budget-gib 0.25 --samples 16 \
  --embedding-cost-model mget --mget-fixed-ms 7.5 --mget-us-per-id 0 \
  --output /tmp/hstu-paired-mget
```

有 miss 才触发一次逻辑后端调用；固定成本对 **整次请求的 missing key 集合** 计一次，既不乘 key 数，也不按 cache shard 重复计费。可用 `--mget-us-per-id` 增加返回规模的线性项。这里执行受控等待，不连接真实 Redis，也没有模拟 Redis 的长尾分布。

## Cache ratio 与动态流量重放

同一流量比较固定配比，每个比例在新进程中创建模型与缓存：

```bash
python3 -s inference/benchmark/cache_benefit/run.py replay \
  --data /path/to/processed_seqs.csv --budget-gib 16 \
  --fractions 0.15,0.35,0.55,0.75,0.95 \
  --warmup 26000 --requests 26000 --window 2000 --traffic original \
  --kv-miss-ms 1.329810 --embedding-us-per-id 0.410387 \
  --output /tmp/hstu-ratios
```

`original` 复用原 `zipf_stream`：预热 seed 与计时 seed 也会重排用户热度。`steady` 固定分布，仅改变抽样；`step` 采用短历史 A→长历史 B→A；`gradual` 采用 A→逐渐混合→B。后两种默认按历史长度将用户划分成约 40%／60% 两组，各组 Zipf 热度不变。每个用户内容始终不变，阶段之间不重置 cache。三阶段按请求序号划分，不代表真实到达时间或并发 QPS。

可用 `--resize-at 2000:0.75,4000:0.35` 做事先指定的容量调整实验。调整耗时计入对应请求 E2E，后续驱逐与回填自然计入；这是脚本化实验，不称为在线算法。

真实全命中基线：

```bash
python3 -s inference/benchmark/cache_benefit/run.py all-hit \
  --data /path/to/processed_seqs.csv --history-cap 4096 --candidates 100 \
  --budget-gib 64 --fraction 0.65 --requests 26000 \
  --output /tmp/hstu-all-hit
```

工具先验证容量足够，预填充后再热身全命中路径，计时阶段逐请求要求 KV 完整 GPU 命中、embedding 无 CPU 回源。仅将等待参数设零不能通过这些检查。

## 输出与口径

每次使用一个新输出目录。主要文件：

| 文件 | 内容 |
| --- | --- |
| `report.md`、`summary.json` | 可审阅的统计、条件收益或连续重放结果 |
| `requests.csv` | 逐请求 E2E、两侧 hit/miss、实际配比、容量／占用和实际执行的模拟等待 |
| `windows.csv` | 重放的分窗 E2E、KV／embedding hit rates |
| `ratios.csv` | 多固定配比重放的统一对比表 |
| `ratio_deltas.csv` | 相邻已测比例的两侧 hit rate 变化与系统净 E2E 收益；不当作单侧 hit 收益 |
| `resizes.json` | 调整耗时、物理驱逐量；当前实现不搬移保留内容 |
| `provenance.json` | 数据、源码、模型与 gin 哈希，硬件／框架版本和成本设置 |

主指标是输入交给请求处理流程后，经过 CPU materialization、H2D、cache 查询与 miss 处理、HSTU forward，到 GPU 预测实际可用的 wall time。模拟等待在 miss 路径执行，**不会事后加到 latency 上**。CSV 加载／日志、网络服务排队、multi-request batch 不包含在本版 E2E 中。

GPU 预算约束 persistent KV + video embedding payload；模型权重、静态 action embedding、activation、索引／临时工作区另计。`torch_gpu_peak_allocated_bytes` 是 Torch allocator 统计，不等于整卡全部使用量。CPU KV 和 SSD 关闭。正常访问是 LRU，resize 时采用物理页驱逐，因此需要观察调整后的恢复损失。

固定比例拥有不同缓存历史，分窗最小值只作为离线参考。它不等于已经执行的动态路径，也不能直接称为动态策略下界。MGET 场景应同时看 embedding **key hit rate** 和 **request all-hit rate**：少量 miss 也可能触发整次调用成本。

正式比较建议改变比例的运行顺序、重复整组，并检查分窗与跨运行波动，避免把硬件／执行时段的变化归因于容量配比。

## 验证

无需 GPU 的测试：

```bash
python3 -m unittest discover -s inference/benchmark/cache_benefit/tests -p 'test_*.py' -v
```

GPU 数据与真实 HSTU 检查：

```bash
python3 -s inference/benchmark/cache_benefit/tests/gpu_smoke.py
```

合作讨论时建议一并提供 `report.md`、`summary.json`、`provenance.json`，并明确数据来源、是否使用模拟等待、命中率的分母、完整 E2E 边界及重复次数。

验证范围与示例见 [VALIDATION.md](VALIDATION.md) 和 [合成数据报告](example/report.md)。
