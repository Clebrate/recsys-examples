> 合成数据功能示例：16 个 synthetic users 中抽取 8 条不同请求，各重复 2 次；history cap=160、C=100、cache 预算 256 MiB、KV fraction=0.55、非零 pattern 参数。KV 额外等待 0.02 ms；MGET 固定成本 0.05 ms + 0.87 µs × missing unique IDs。
>
> 这些小样本数字用于展示报告格式和计费语义，不能作为生产收益或在线算法结论。负收益保留；真实业务结论需用目标流量和独立重复运行验证。

# HSTU cache benefit benchmark

Full-request E2E includes CPU request preparation, H2D, actual cache handling, executed synthetic waits, HSTU and completion synchronization. One request at a time; no service queue or real Redis measurement.

## Same-request four-state measurements

First character is KV state, second is embedding state. Embedding M means all unique keys initially miss; repeated occurrences within that request can hit after their first fill.

| State | Mean ms | p50 ms | p95 ms | KV request hit | Embedding key hit | Embedding request hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HH | 3.7049 | 3.6872 | 3.8818 | 100.0% | 100.0% | 100.0% |
| MH | 3.6099 | 3.6408 | 3.7455 | 0.0% | 100.0% | 100.0% |
| HM | 4.1922 | 4.2016 | 4.3296 | 100.0% | 1.5% | 0.0% |
| MM | 4.1196 | 4.1520 | 4.2928 | 0.0% | 1.5% | 0.0% |

## Hit savings: miss E2E minus hit E2E

| Conditional saving | Mean ms/request | Request-cluster bootstrap 95% interval, ms |
| --- | ---: | --- |
| kv_saving_given_embedding_hit | -0.0950 | -0.1487 to -0.0254 |
| kv_saving_given_embedding_miss | -0.0726 | -0.1173 to -0.0332 |
| embedding_saving_given_kv_hit | 0.4873 | 0.4380 to 0.5474 |
| embedding_saving_given_kv_miss | 0.5097 | 0.4494 to 0.5712 |
| interaction | 0.0224 | -0.0351 to 0.0746 |

Interaction = L(MM) − L(HM) − L(MH) + L(HH). Negative measured savings are retained. Intervals describe sampled requests with repeats clustered together; use independent runs to assess run/hardware variation.

Cache warmup/invalidation fixtures are outside each timed counterfactual. This isolates conditional benefits and does not simulate natural online eviction/recovery.

Configuration, source/model hashes and cost semantics are in `provenance.json`. No trained checkpoint is loaded. This experimental cache container reuses the reference HSTU kernels; it is not an NVE/FlexKV public resize API.
