# Blackwell Bug Notes (HSTU + FlexKV Benchmark)

本文记录在 Blackwell (SM 10.0) 上跑 `examples/hstu/inference/benchmark/inference_benchmark.py` 时已遇到的问题、原因判断和可行修复方案。

## 环境与范围

- 目标平台: Blackwell (B200, `sm_100`)
- 相关路径:
  - `examples/hstu/inference/benchmark/inference_benchmark.py`
  - `examples/hstu/modules/paged_hstu_infer_layer.py`
  - `examples/hstu/ops/pt_ops/torch_addmm.py`
  - `corelib/dynamicemb/setup.py`
- 参考日志:
  - `benchmark_with_FlexKV/blackwell/20260428_070714/baseline_nokvcache_bs1.log`
  - `benchmark_with_FlexKV/blackwell/20260428_074326/baseline_nokvcache_bs1.log`
  - `terminals/7.txt` (本地会话中的报错)

---

## Bug 1: dynamicemb 无法在 Blackwell 执行

### 现象

- 报错:
  - `RuntimeError: cudaCheckError() ... no kernel image is available for execution on the device`
- 示例日志:
  - `blackwell/20260428_070714/baseline_nokvcache_bs1.log`

### 原因判断

- `dynamicemb` 扩展未包含 `sm_100` 的 CUDA kernel image。

### 可行修复

1. 在 `corelib/dynamicemb/setup.py` 增加 Blackwell 编译目标:
   - `-gencode arch=compute_100,code=sm_100`
   - `-gencode arch=compute_100,code=compute_100`
2. 重新安装/替换 dynamicemb。
3. 用 `cuobjdump` 或等效方法确认 `.so` 中存在 `sm_100`。

### 当前状态

- 已有后续日志不再出现该报错，说明该问题已基本被绕开/修复。

---

## Bug 2: HSTU 层对 SM 10 直接拒绝

### 现象

- 报错:
  - `ValueError: Unsupported SM major version: 10`
- 示例日志:
  - `blackwell/20260428_074326/baseline_nokvcache_bs1.log`

### 原因判断

- `paged_hstu_infer_layer.py` 中分支只覆盖了 `sm == 8` 与 `sm == 9`，未覆盖 `sm == 10`。

### 可行修复

有两种思路:

1. 快速放开:
   - 将 `sm == 9` 改成 `sm >= 9`，先让 Blackwell 复用 PyTorch fallback 路径。
2. 更稳妥:
   - 显式拆分:
     - `sm == 9` 保持原逻辑
     - `sm == 10` 单独分支，绑定 Blackwell 专用实现/兼容 wrapper

### 当前状态

- 仅放开到 `sm >= 9` 虽然能绕过该错误，但会触发后续参数兼容问题（见 Bug 3）。

---

## Bug 3: Blackwell fallback 调用签名不兼容

### 现象

- 报错:
  - `TypeError: torch_addmm_silu_fwd() got an unexpected keyword argument 'keep_unfused_out'`
- 示例日志:
  - `terminals/7.txt`

### 原因判断

- Blackwell 被路由到 `addmm_silu_impl = torch_addmm_silu_fwd`。
- 但调用方按 Triton 风格传参:
  - `keep_unfused_out`
  - `silu_out`
  - `out`
- 当前 `torch_addmm_silu_fwd` 不接收这些参数，触发 `TypeError`。

### 可行修复

两种可选:

1. 改 `torch_addmm_silu_fwd` 签名（全局兼容）
   - 在 `examples/hstu/ops/pt_ops/torch_addmm.py` 扩展参数并兼容 `out/silu_out/keep_unfused_out`。
2. 不改通用 PT op，按架构分流（更保守）
   - 在 `paged_hstu_infer_layer.py`:
     - `sm == 9` 继续用现有 `torch_addmm_silu_fwd`
     - `sm == 10` 走单独 wrapper（wrapper 接收 Triton 风格参数，再调用 `torch_addmm_silu_fwd` 做转换）

### 建议优先级

- 短期建议: 采用方案 2（影响面最小，易回滚）。
- 中期建议: 评估方案 1（减少多处分支维护成本）。

### 当前状态

- 待修复（已确认根因与可行方案）。

---

## 建议的后续验证清单

- `baseline_nokvcache` 在 Blackwell `bs=1/2/4/8` 全通过。
- `kvcache_nop` 在 Blackwell 最少 `bs=1,4` 通过。
- `kvcache_flexkv_direct` 在 Blackwell 最少 `bs=1,4` 通过。
- 如需对齐生产路径，再补 `kvcache_flexkv_server_client`。
- 每个 case 记录:
  - `Total time(ms)`
  - 是否有 `Traceback`
  - 是否有 FlexKV init/ready timeout。

