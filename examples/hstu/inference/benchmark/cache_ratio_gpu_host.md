Total cache size: GPU 上的 KV cache 和 embedding cache 总量固定为 16 / 32 / 64 GiB。

Workload: 取 KuaiRand-27K 前半段，序列末尾 100 个 video 当作候选，再往前当作历史。因此 KV 全部命中时所需要的 cache size = 48.72 GiB，embedding 全部命中时所需要的 cache size = 3.08 GiB。
Model: `kuairand_1k_inference_ranking.gin`：N = 3，H = 4，d = 128，d_h = 512，s = 1024，t = 2048，c = 100。

Cache ratio: 0.05 - 0.95, 步长 = 0.05

# Phase1: KV miss uses recompute，embedding miss use CPU parameter table.

KV miss 直接在 GPU 上重算。embedding miss 进 CPU 参数表。

每张图的左轴是平均延迟，右轴是命中率。蓝线是 GPU KV token 命中率。橙线是 embedding 命中率。

对于 KV 而言，未命中是重算，命中是从 GPU 读 KV。

$$
L_{\mathrm{miss}}-L_{\mathrm{hit}}=0.060-0.00788=0.052\ \mathrm{ms}
$$

对于 embedding 而言，一个 id 未命中走 PCIe，命中走 GPU。

$$
L_{\mathrm{miss}}-L_{\mathrm{hit}}=0.0325-0.00128=0.031\ \mu\mathrm{s}
$$

## Total Cache size = 16 GiB

![16 GiB GPU 平均延迟和命中率](figures/gpu_16_latency_hit.png)

平均延迟随着 KV cache ratio 的增加而呈下降趋势。当 KV 比例 ≤ 0.80 时，embedding cache ≥ 16 × (1 − 0.80) = 3.20 GiB，embedding 命中率是 1。

因此，当 KV cache ratio < 0.80 时，embedding 的 hit rate = 1，latency 只和 KV 有关，随着 KV cache 增大，latency 逐渐减小。

当 KV cache ratio > 0.80 时，embedding hit rate < 1（KV cache ratio = 0.85、0.90、0.95 降到 0.830、0.710、0.555）。一次 embedding miss 只有 0.031 µs，远小于 KV 命中省下的 0.31 ms。因此随着 KV cache ratio 增大，latency 降低。

## Total Cache size = 32 GiB

![32 GiB GPU 平均延迟和命中率](figures/gpu_32_latency_hit.png)

平均延迟随着 KV cache ratio 的增加而呈下降趋势。当 KV 比例 ≤ 0.90 时，embedding cache ≥ 32 × (1 − 0.90) = 3.20 GiB，embedding 命中率是 1。

因此，当 KV cache ratio < 0.90 时，embedding 的 hit rate = 1，latency 只和 KV 有关，随着 KV cache 增大，latency 逐渐减小。

当 KV cache ratio > 0.90 时，只有 0.95 这一格，embedding hit rate 降到 0.691。一次 embedding miss 只有 0.031 µs，小于 KV 命中省下的 0.51 ms。

## Total Cache size = 64 GiB

![64 GiB GPU 平均延迟和命中率](figures/gpu_rerun_64_mean.png)

首先 KV cache ratio 从 0.05 - 0.95，embedding cache > 3.2GiB, embedding 是一定命中。当 KV cache ratio > 0.8 时，KV 开始全部命中。
因此，平均延迟随着 KV cache ratio 的增加而下降，KV 全中之后逐渐平缓。

# Phase2: KVCache and embedding cache in GPU，KV miss first CPU load and recompute，embedding miss use CPU parameter table.

GPU 上的划分和 Phase1 相同。另外加一块固定的 32 GiB CPU KV cache (CPU hit rate： ～0.4)。KV 先查 GPU，没有再查这块 CPU cache，两边都没有才重算。embedding miss 仍进 CPU 参数表。


图的左轴仍是平均延迟，右轴是 GPU KV token 命中率和 embedding 命中率。

对于 KV 而言，三条路径是

$$
\begin{aligned}
L_{\mathrm{GPU}} &= 7.88\ \mu\mathrm{s} \\
L_{\mathrm{CPU}} &= 0.208\ \mathrm{ms} \\
L_{\mathrm{recompute}} &= 0.060\ \mathrm{ms}
\end{aligned}
$$

两两相减：

$$
\begin{aligned}
L_{\mathrm{CPU}}-L_{\mathrm{GPU}} &= 0.208-0.00788=0.200\ \mathrm{ms} \\
L_{\mathrm{recompute}}-L_{\mathrm{GPU}} &= 0.060-0.00788=0.052\ \mathrm{ms} \\
L_{\mathrm{recompute}}-L_{\mathrm{CPU}} &= 0.060-0.208=-0.148\ \mathrm{ms}
\end{aligned}
$$

第三行里 miss 是重算，hit 是 CPU 命中，所以 $L_{\mathrm{miss}}-L_{\mathrm{hit}}$ 是负的。CPU 命中最慢，GPU 命中最快，重算在中间。

对于 embedding 而言，和 Phase1 相同。

$$
L_{\mathrm{miss}}-L_{\mathrm{hit}}=0.031\ \mu\mathrm{s}
$$


## Total Cache size = 16 GiB

![16 GiB GPU 预算加 32 GiB CPU KV](figures/gpu_host32_16_marked.png)

当 KV 比例 ≤ 0.80 时，embedding cache ≥ 16 × (1 − 0.80) = 3.20 GiB，命中率是 1。GPU KV 最大 12.80 GiB，小于 48.72 GiB，没有 KV 全部在 GPU 命中。
KV cache ratio = 0.05 时：GPU 命中 0.012，CPU 命中 0.430，重算 0.558。
KV cache ratio = 0.80：GPU 命中 0.256，CPU 命中 0.200，重算 0.544。

因此，随着 KV Cache ratio 的增加，GPU 命中率增加，GPU latency < CPU latency，因此平均延迟下降。

当 KV 比例 > 0.80 时，embedding hit rate 降到 0.559。随着 KV Cache ratio 的增加，GPU 命中率增加，embedding hit rate 下降，但是 KV cache 收益 > embedding 收益，所以 latency 同样下降。0.40 的 4.471 ms 和 0.50 的 4.478 ms 是抖动。最低在 0.95，4.479 ms。

## Total Cache size = 32 GiB

![32 GiB GPU 预算加 32 GiB CPU KV](figures/gpu_host32_32_mean.png)

当 KV 比例 ≤ 0.90 时，embedding cache ≥ 32 × (1 − 0.90) = 3.20 GiB，命中率是 1。GPU KV 最大 28.80 GiB，小于 48.72 GiB，没有 KV 全部在 GPU 命中。
KV cache ratio = 0.05 时：GPU 命中 0.036，CPU 命中 0.415，重算 0.549。
KV cache ratio = 0.90：GPU 命中 0.466，CPU 命中 0.027，重算 0.507。

因此，随着 KV Cache ratio 的增加，GPU 命中率增加，GPU latency < CPU latency，因此平均延迟下降。

当 KV 比例 > 0.90 时，只有 0.95 这一格，embedding hit rate 降到 0.692。随着 KV Cache ratio 的增加，GPU 命中率增加，embedding hit rate 下降，但是 KV cache 收益 > embedding 收益，所以 latency 同样下降。

## Total Cache size = 64 GiB


![64 GiB GPU 预算加 32 GiB CPU KV](figures/gpu_host32_64_latency_hit.png)

当 KV 比例 ≤ 0.95 时，embedding cache ≥ 64 × (1 − 0.95) = 3.20 GiB，命中率是 1。KV 全部在 GPU 命中从比例 0.80 开始，GPU KV = 51.20 GiB，大于 48.72 GiB。
KV cache ratio = 0.05 时：GPU 命中 0.073，CPU 命中 0.378，重算 0.549。
KV cache ratio = 0.80：GPU 命中 1，CPU 命中 0，重算 0。

因此，随着 KV Cache ratio 的增加，GPU 命中率增加，GPU latency < CPU latency，因此平均延迟下降。比例 ≥ 0.80 的 0.80、0.85、0.90、0.95 都是最低这一档，延迟在 3.935–4.044 ms。KV 已经全部在 GPU 命中，embedding hit rate 仍是 1。
