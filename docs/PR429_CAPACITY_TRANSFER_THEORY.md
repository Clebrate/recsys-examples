# PR429 FlexKV 容量、计算与传输理论估算

## 结论

- 推荐 `cpu_cache_gb=32`、`ssd_cache_gb=200`（实际容量下限约 56 GiB，保守可配 96–128 GiB）。
- 不建议直接把 CPU cache 加到 64 GiB：它不会降低 SSD→CPU 或 CPU→GPU 延迟，反而会让 scenario3 生成约两倍 pressure 数据，并锁页更多主存。
- 本轮 L40S 实测 CPU→GPU 为 25–26 GiB/s，已接近 PCIe 4.0 x16 单向理论上限 31.5 GB/s。
- 本轮本地 SSD→CPU 实测约 2.9–3.4 GiB/s，是 scenario3 的主要瓶颈。
- `hl4096` 失败不是容量公式错误，而是双进程残留占用约 20 GiB GPU 显存；干净节点应先确认 `nvidia-smi` 无残留。

## KV 容量公式

模型参数：8 layers、K/V 两份、4 heads、head_dim=256、BF16（2 bytes）、page_size=32。

```text
每 token KV bytes = 8 × 2 × 4 × 256 × 2 = 32 KiB
每 block bytes    = 32 tokens × 32 KiB = 1 MiB
每 user blocks    = history_len × 2 / 32 = history_len / 16
每 user KV        = history_len / 16 MiB
```

| history_len | attn KV tokens | 每 user KV | bs=1 | bs=2 | bs=4 | bs=8 |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1,280 | 32 MiB | 32 MiB | 64 MiB | 128 MiB | 256 MiB |
| 1024 | 2,304 | 64 MiB | 64 MiB | 128 MiB | 256 MiB | 512 MiB |
| 2048 | 4,352 | 128 MiB | 128 MiB | 256 MiB | 512 MiB | 1 GiB |
| 4096 | 8,448 | 256 MiB | 256 MiB | 512 MiB | 1 GiB | 2 GiB |

10 个 timed iteration 使用不同 user，因此最坏 `hl4096, bs8` target set：

```text
10 × 8 × 256 MiB = 20 GiB
```

32 GiB CPU cache 留出约 12 GiB 给临时 block、并发 PUT、锁页和 allocator 余量。scenario3 使用 32 GiB CPU 时，各 history_len 的 target + pressure 总生成量约为：

| history_len | pressure users | pressure KV | 10×bs8 targets | 总生成量 |
|---:|---:|---:|---:|---:|
| 512 | 1,038 | 32.44 GiB | 2.50 GiB | 34.94 GiB |
| 1024 | 526 | 32.88 GiB | 5.00 GiB | 37.88 GiB |
| 2048 | 270 | 33.75 GiB | 10.00 GiB | 43.75 GiB |
| 4096 | 142 | 35.50 GiB | 20.00 GiB | 55.50 GiB |

因此 SSD 96 GiB 已有余量；保留 200 GiB 可避免边界条件，但不会提升传输速度。

## 理论计算量（粗略 attention 上界）

令 `N = 2 × history_len + 256`、hidden heads 总维度 `D=1024`、layers=8。

```text
no_cache attention FLOPs ≈ 4 × layers × D × N²
s1 cached-tail FLOPs     ≈ 4 × layers × D × 256 × N
```

| history_len | N | no_cache / sample | s1 cached-tail / sample | L40S BF16 峰值时间（no_cache） |
|---:|---:|---:|---:|---:|
| 512 | 1,280 | 53.7 GFLOP | 10.7 GFLOP | 0.15 ms |
| 1024 | 2,304 | 173.9 GFLOP | 19.3 GFLOP | 0.48 ms |
| 2048 | 4,352 | 620.6 GFLOP | 36.5 GFLOP | 1.71 ms |
| 4096 | 8,448 | 2.34 TFLOP | 70.9 GFLOP | 6.46 ms |

峰值时间按 L40S dense BF16 362.05 TFLOP/s 计算，仅是不可达到的下界；实际还包含投影、MLP、embedding、kernel launch、同步和低利用率小矩阵。

## Transfer 下界与实测模型

L40S 为 PCIe 4.0 x16：单向理论约 31.5 GB/s；本轮实测 H2D 约 25.7 GiB/s。本地 SSD→CPU 实测取 3.2 GiB/s。

```text
s2 H2D 实测估算 = KV_GiB / 25.7 GiB/s
s3 串行估算     = KV_GiB / 3.2 + KV_GiB / 25.7
```

| history_len | bs | KV transfer | H2D 理论下界 | H2D 实测估算 | SSD→CPU→GPU 实测估算 |
|---:|---:|---:|---:|---:|---:|
| 512 | 1/2/4/8 | 0.031/0.062/0.125/0.250 GiB | 0.99/1.98/3.97/7.94 ms | 1.22/2.43/4.86/9.73 ms | 10.98/21.96/43.93/87.85 ms |
| 1024 | 1/2/4/8 | 0.062/0.125/0.250/0.500 GiB | 1.98/3.97/7.94/15.87 ms | 2.43/4.86/9.73/19.46 ms | 21.96/43.93/87.85/175.71 ms |
| 2048 | 1/2/4/8 | 0.125/0.250/0.500/1.000 GiB | 3.97/7.94/15.87/31.75 ms | 4.86/9.73/19.46/38.91 ms | 43.93/87.85/175.71/351.41 ms |
| 4096 | 1/2/4/8 | 0.250/0.500/1.000/2.000 GiB | 7.94/15.87/31.75/63.49 ms | 9.73/19.46/38.91/77.82 ms | 87.85/175.71/351.41/702.82 ms |

当前部分 s3 平均值低于这个物理传输估算，说明并非每个 iteration 都是纯 SSD hit；新节点复测时应逐 iteration 验证 `DISK2H + H2D`。

## 新节点复测

### 申请 L40S 节点

```bash
crun -q 'gpu.product_name=*L40S* and cpu.arch=x86_64' \
  --gpus=1 -s bash -t15:00:00 -i
```

### 在 compute node 启动容器

```bash
docker rm -f pr429_cpu32 2>/dev/null || true
docker run -it \
  --name pr429_cpu32 \
  --gpus all \
  --shm-size=16g \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --cap-add SYS_PTRACE \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/scratch.noliu_gpu:/home/scratch.noliu_gpu \
  -u 0:0 \
  -e HOME=/root \
  -e USER=root \
  -w /home/scratch.noliu_gpu/pr429-verify \
  gitlab-master.nvidia.com:5005/devtech-compute/distributed-recommender:inference_latest \
  bash
```

### 容器内补跑缺失/损坏项

```bash
cd /home/scratch.noliu_gpu/pr429-verify
nvidia-smi
free -h
df -h /tmp

export FLEXKV_SSD_CACHE_BASE=/tmp/nvidia-mp
nohup bash scripts/rerun_incomplete_pr429_sweep.sh \
  > /tmp/pr429_cpu32_rerun.log 2>&1 &
echo "pid=$!"
tail -f /tmp/pr429_cpu32_rerun.log
```

### 完整 CPU32 对照复测（写入新目录，不覆盖旧结果）

```bash
cd /home/scratch.noliu_gpu/pr429-verify
export FLEXKV_SSD_CACHE_BASE=/tmp/nvidia-mp
export OUT_DIR=/home/scratch.noliu_gpu/nsys_pr429_timed_sweep_cpu32
nohup bash scripts/run_sweep_in_container.sh \
  > /tmp/pr429_cpu32_full.log 2>&1 &
echo "pid=$!"
tail -f /tmp/pr429_cpu32_full.log
```
