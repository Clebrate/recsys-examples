# PR429 Timed Sweep Analysis

所有指标仅来自 **timed NVTX range**：`scenario{N}_timed_run_*` / `no_cache_timed_run_*`。

## Sequence length 说明

| 参数 | 含义 |
|------|------|
| `history_len=1024` | 单流 item/action history 各 1024 token |
| only_onboard timed attention KV | `history_len*2 + 256` = **2304** |
| main flexkv timed（带 append） | `(history+append)*2 + 256` = **4352** |

对比 kvcache 收益请使用 **only_onboard + no_cache** 同一 timed shape。

## 全量 timed 数据

| label | scenario | bs | history_len | attn_kv | only_onboard | skip_offload | avg_ms | min_ms | max_ms | HtoD_MB | DtoH_MB | onboard_wait_ms |
|-------|----------|---:|------------:|--------:|:------------:|:------------:|-------:|-------:|-------:|--------:|--------:|----------------:|
| `no_cache_only_onboard_hl1024_bs1` | no_cache | 1 | 1024 | 2304 | Y | ? | 16.60 | 16.20 | 18.92 | 0.18 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl1024_bs2` | no_cache | 2 | 1024 | 2304 | Y | ? | 16.50 | 16.18 | 18.01 | 0.35 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl1024_bs4` | no_cache | 4 | 1024 | 2304 | Y | ? | 16.76 | 16.51 | 18.03 | 0.70 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl1024_bs8` | no_cache | 8 | 1024 | 2304 | Y | ? | 17.73 | 17.38 | 19.86 | 1.41 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl2048_bs1` | no_cache | 1 | 2048 | 4352 | Y | ? | 16.66 | 16.01 | 18.06 | 0.33 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl2048_bs2` | no_cache | 2 | 2048 | 4352 | Y | ? | 16.69 | 16.20 | 18.14 | 0.66 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl2048_bs4` | no_cache | 4 | 2048 | 4352 | Y | ? | 18.57 | 18.08 | 19.44 | 1.33 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl2048_bs8` | no_cache | 8 | 2048 | 4352 | Y | ? | 34.80 | 34.27 | 35.52 | 2.66 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl4096_bs1` | no_cache | 1 | 4096 | 8448 | Y | ? | 16.56 | 16.26 | 17.86 | 0.64 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl4096_bs2` | no_cache | 2 | 4096 | 8448 | Y | ? | 22.77 | 21.99 | 23.45 | 1.29 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl4096_bs4` | no_cache | 4 | 4096 | 8448 | Y | ? | 44.17 | 43.00 | 45.45 | 2.58 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl4096_bs8` | no_cache | 8 | 4096 | 8448 | Y | ? | 87.66 | 83.71 | 89.65 | 5.16 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl512_bs1` | no_cache | 1 | 512 | 1280 | Y | ? | 4.96 | 4.84 | 5.78 | 0.10 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl512_bs2` | no_cache | 2 | 512 | 1280 | Y | ? | 16.34 | 16.07 | 17.77 | 0.20 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl512_bs4` | no_cache | 4 | 512 | 1280 | Y | ? | 16.67 | 16.41 | 17.91 | 0.39 | 0.00 | 0.00 |
| `no_cache_only_onboard_hl512_bs8` | no_cache | 8 | 512 | 1280 | Y | ? | 17.58 | 16.90 | 19.51 | 0.78 | 0.00 | 0.00 |
| `only_onboard_s1_gpu_hl1024_bs1` | s1_gpu | 1 | 1024 | 2304 | Y | ? | 7.75 | 6.53 | 17.52 | 0.36 | 640.01 | 0.00 |
| `only_onboard_s1_gpu_hl1024_bs2` | s1_gpu | 2 | 1024 | 2304 | Y | ? | 9.19 | 7.43 | 22.80 | 0.72 | 1280.01 | 0.00 |
| `only_onboard_s1_gpu_hl1024_bs4` | s1_gpu | 4 | 1024 | 2304 | Y | ? | 11.06 | 8.72 | 31.66 | 1.43 | 2560.02 | 0.00 |
| `only_onboard_s1_gpu_hl1024_bs8` | s1_gpu | 8 | 1024 | 2304 | Y | ? | 26.60 | 22.82 | 58.95 | 2.86 | 5120.05 | 0.00 |
| `only_onboard_s1_gpu_hl2048_bs1` | s1_gpu | 1 | 2048 | 4352 | Y | ? | 8.21 | 6.63 | 21.70 | 0.68 | 1280.01 | 0.00 |
| `only_onboard_s1_gpu_hl2048_bs2` | s1_gpu | 2 | 2048 | 4352 | Y | ? | 9.98 | 7.36 | 31.97 | 1.35 | 2560.02 | 0.00 |
| `only_onboard_s1_gpu_hl2048_bs4` | s1_gpu | 4 | 2048 | 4352 | Y | ? | 13.05 | 8.78 | 49.06 | 2.70 | 5120.04 | 0.00 |
| `only_onboard_s1_gpu_hl2048_bs8` | s1_gpu | 8 | 2048 | 4352 | Y | ? | 29.18 | 22.96 | 81.25 | 5.40 | 10240.09 | 0.00 |
| `only_onboard_s1_gpu_hl512_bs1` | s1_gpu | 1 | 512 | 1280 | Y | ? | 7.07 | 6.57 | 8.76 | 0.20 | 320.00 | 0.00 |
| `only_onboard_s1_gpu_hl512_bs2` | s1_gpu | 2 | 512 | 1280 | Y | ? | 8.28 | 7.20 | 17.24 | 0.40 | 640.01 | 0.00 |
| `only_onboard_s1_gpu_hl512_bs4` | s1_gpu | 4 | 512 | 1280 | Y | ? | 9.88 | 8.36 | 22.27 | 0.80 | 1280.01 | 0.00 |
| `only_onboard_s1_gpu_hl512_bs8` | s1_gpu | 8 | 512 | 1280 | Y | ? | 24.56 | 22.40 | 38.26 | 1.59 | 2560.03 | 0.00 |
| `only_onboard_s2_cpu_hl1024_bs1` | s2_cpu | 1 | 1024 | 2304 | Y | ? | 9.81 | 8.64 | 18.13 | 0.36 | 640.01 | 13.25 |
| `only_onboard_s2_cpu_hl1024_bs2` | s2_cpu | 2 | 1024 | 2304 | Y | ? | 12.71 | 11.50 | 21.39 | 0.72 | 1280.01 | 39.63 |
| `only_onboard_s2_cpu_hl1024_bs4` | s2_cpu | 4 | 1024 | 2304 | Y | ? | 18.97 | 17.88 | 27.17 | 1.43 | 2560.02 | 87.26 |
| `only_onboard_s2_cpu_hl1024_bs8` | s2_cpu | 8 | 1024 | 2304 | Y | ? | 41.82 | 41.36 | 44.16 | 2.86 | 5120.05 | 184.58 |
| `only_onboard_s2_cpu_hl2048_bs1` | s2_cpu | 1 | 2048 | 4352 | Y | ? | 12.29 | 11.10 | 20.58 | 0.68 | 1280.01 | 38.04 |
| `only_onboard_s2_cpu_hl2048_bs2` | s2_cpu | 2 | 2048 | 4352 | Y | ? | 17.84 | 16.75 | 26.08 | 1.35 | 2560.02 | 92.74 |
| `only_onboard_s2_cpu_hl2048_bs4` | s2_cpu | 4 | 2048 | 4352 | Y | ? | 29.43 | 28.08 | 37.60 | 2.70 | 5120.04 | 192.50 |
| `only_onboard_s2_cpu_hl2048_bs8` | s2_cpu | 8 | 2048 | 4352 | Y | ? | 64.58 | 63.34 | 68.17 | 5.40 | 10240.09 | 388.35 |
| `only_onboard_s2_cpu_hl512_bs1` | s2_cpu | 1 | 512 | 1280 | Y | ? | 8.42 | 7.71 | 11.07 | 0.20 | 320.00 | 3.53 |
| `only_onboard_s2_cpu_hl512_bs2` | s2_cpu | 2 | 512 | 1280 | Y | ? | 10.00 | 8.92 | 18.39 | 0.40 | 640.01 | 12.17 |
| `only_onboard_s2_cpu_hl512_bs4` | s2_cpu | 4 | 512 | 1280 | Y | ? | 13.49 | 12.36 | 21.81 | 0.80 | 1280.01 | 35.31 |
| `only_onboard_s2_cpu_hl512_bs8` | s2_cpu | 8 | 512 | 1280 | Y | ? | 32.36 | 31.58 | 35.48 | 1.59 | 2560.03 | 83.88 |
| `only_onboard_s3_ssd_hl1024_bs1` | s3_ssd | 1 | 1024 | 2304 | Y | ? | 31.38 | 28.45 | 42.36 | 2.65 | 8832.05 | 235.76 |
| `only_onboard_s3_ssd_hl1024_bs2` | s3_ssd | 2 | 1024 | 2304 | Y | ? | 55.55 | 52.93 | 65.70 | 3.00 | 9472.06 | 460.67 |
| `only_onboard_s3_ssd_hl1024_bs4` | s3_ssd | 4 | 1024 | 2304 | Y | ? | 95.93 | 20.59 | 167.59 | 3.72 | 10752.07 | 804.71 |
| `only_onboard_s3_ssd_hl1024_bs8` | s3_ssd | 8 | 1024 | 2304 | Y | ? | 88.31 | 22.58 | 229.52 | 5.15 | 11264.09 | 576.73 |
| `only_onboard_s3_ssd_hl2048_bs1` | s3_ssd | 1 | 2048 | 4352 | Y | ? | 55.67 | 51.84 | 67.26 | 5.00 | 17536.09 | 478.80 |
| `only_onboard_s3_ssd_hl2048_bs2` | s3_ssd | 2 | 2048 | 4352 | Y | ? | 97.05 | 20.43 | 174.05 | 5.67 | 18944.10 | 822.53 |
| `only_onboard_s3_ssd_hl2048_bs4` | s3_ssd | 4 | 2048 | 4352 | Y | ? | 82.79 | 21.92 | 216.46 | 7.02 | 19456.12 | 570.58 |
| `only_onboard_s3_ssd_hl512_bs1` | s3_ssd | 1 | 512 | 1280 | Y | ? | 19.34 | 18.06 | 25.10 | 1.47 | 4416.03 | 122.05 |
| `only_onboard_s3_ssd_hl512_bs2` | s3_ssd | 2 | 512 | 1280 | Y | ? | 32.77 | 31.51 | 37.27 | 1.67 | 4736.04 | 235.93 |
| `only_onboard_s3_ssd_hl512_bs4` | s3_ssd | 4 | 512 | 1280 | Y | ? | 55.75 | 53.05 | 64.81 | 2.07 | 5376.04 | 466.68 |
| `only_onboard_s3_ssd_hl512_bs8` | s3_ssd | 8 | 512 | 1280 | Y | ? | 84.31 | 23.02 | 120.07 | 2.86 | 6656.05 | 561.88 |

## 对齐对比（timed avg_ms，相对 no_cache 的 ratio）

### history_len=512 (attn_kv=1280)

| bs | no_cache | s1_gpu | s2_cpu | s3_ssd | s1/nc | s2/nc | s3/nc | speedup s1 | speedup s2 | speedup s3 |
|---:|---------:|-------:|-------:|-------:|------:|------:|------:|-----------:|-----------:|-----------:|
| 1 | 4.96 | 7.07 | 8.42 | 19.34 | 1.43x | 1.70x | 3.90x | 0.70x | 0.59x | 0.26x |
| 2 | 16.34 | 8.28 | 10.00 | 32.77 | 0.51x | 0.61x | 2.01x | 1.97x | 1.63x | 0.50x |
| 4 | 16.67 | 9.88 | 13.49 | 55.75 | 0.59x | 0.81x | 3.34x | 1.69x | 1.24x | 0.30x |
| 8 | 17.58 | 24.56 | 32.36 | 84.31 | 1.40x | 1.84x | 4.80x | 0.72x | 0.54x | 0.21x |

### history_len=1024 (attn_kv=2304)

| bs | no_cache | s1_gpu | s2_cpu | s3_ssd | s1/nc | s2/nc | s3/nc | speedup s1 | speedup s2 | speedup s3 |
|---:|---------:|-------:|-------:|-------:|------:|------:|------:|-----------:|-----------:|-----------:|
| 1 | 16.60 | 7.75 | 9.81 | 31.38 | 0.47x | 0.59x | 1.89x | 2.14x | 1.69x | 0.53x |
| 2 | 16.50 | 9.19 | 12.71 | 55.55 | 0.56x | 0.77x | 3.37x | 1.80x | 1.30x | 0.30x |
| 4 | 16.76 | 11.06 | 18.97 | 95.93 | 0.66x | 1.13x | 5.72x | 1.52x | 0.88x | 0.17x |
| 8 | 17.73 | 26.60 | 41.82 | 88.31 | 1.50x | 2.36x | 4.98x | 0.67x | 0.42x | 0.20x |

### history_len=2048 (attn_kv=4352)

| bs | no_cache | s1_gpu | s2_cpu | s3_ssd | s1/nc | s2/nc | s3/nc | speedup s1 | speedup s2 | speedup s3 |
|---:|---------:|-------:|-------:|-------:|------:|------:|------:|-----------:|-----------:|-----------:|
| 1 | 16.66 | 8.21 | 12.29 | 55.67 | 0.49x | 0.74x | 3.34x | 2.03x | 1.36x | 0.30x |
| 2 | 16.69 | 9.98 | 17.84 | 97.05 | 0.60x | 1.07x | 5.82x | 1.67x | 0.94x | 0.17x |
| 4 | 18.57 | 13.05 | 29.43 | 82.79 | 0.70x | 1.59x | 4.46x | 1.42x | 0.63x | 0.22x |
| 8 | 34.80 | 29.18 | 64.58 | n/a | 0.84x | 1.86x | n/a | 1.19x | 0.54x | n/a |

### history_len=4096 (attn_kv=8448)

| bs | no_cache | s1_gpu | s2_cpu | s3_ssd | s1/nc | s2/nc | s3/nc | speedup s1 | speedup s2 | speedup s3 |
|---:|---------:|-------:|-------:|-------:|------:|------:|------:|-----------:|-----------:|-----------:|
| 1 | 16.56 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2 | 22.77 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 4 | 44.17 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 8 | 87.66 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## 数据完整性

- 目标 64 项（4 scenarios × 4 history_len × 4 batch_size）。
- 本报告只收录 **含 timed NVTX range** 的 sqlite。
- 已知缺口：`only_onboard_s3_ssd_hl2048_bs8` 缺失；多数 `hl4096` flexkv sqlite 在双进程冲突时导出损坏（无 `NVTX_EVENTS`）。

## 解读要点（基于本轮 only_onboard sweep）

- **有效记录 51/64**：缺 `s3_ssd hl2048 bs8`；`hl4096` 的 s1/s2/s3 sqlite 无有效 timed NVTX。
- **s1_gpu**：中等配置（如 hl1024/2048、bs≤4）通常快于 no_cache（约 **1.4–2.1x**）；短序列 hl512 bs1 和部分 bs=8 点会被固定开销抵消。
- **s2_cpu**：通常介于 s1 与 s3 之间；batch 放大后 H2D 成为主导，部分点慢于 no_cache。
- **s3_ssd**：当前有效点全部慢于 no_cache；大 batch 的 timed iteration 方差很大，需在干净单进程节点复测。
- **speedup = no_cache / scenario**（越大越好）；**sX/nc** 是 latency ratio（越小越好）。

更新本报告：

```bash
python3 scripts/analyze_pr429_timed_sweep.py
```
