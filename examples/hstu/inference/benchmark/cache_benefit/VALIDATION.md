# Validation record

2026-11-09，在 RTX PRO 6000 Blackwell Server GPU、干净的仓库工作目录中验证。HSTU／NVE 模块来自本分支的已提交版本，未依赖正在进行的研究实验里的临时模块修改。

| 检查 | 结果 |
| --- | --- |
| CPU 单元测试 | 7 项通过：容量／失效不变量、MGET 整次成本、条件收益与交互、CSV 输入、可复现流量 |
| GPU 数据保留与失效 | 50,000 个非零 embedding key、重复 key 回填、KV 页面、两个 resize 方向通过 |
| 四状态 HSTU | HH／MH／HM／MM 实际命中状态、miss 数、输出一致性通过 |
| Paired CLI | 8 个请求 × 2 次重复 × 4 状态通过 |
| Ratio CLI | 两个比例各 90 条计时请求，模型与请求流哈希一致 |
| Resize CLI | 90 条计时请求，执行两次容量调整通过 |
| All-hit CLI | 64 用户预填充及全命中路径热身后，60 条计时请求 KV／embedding 全命中，后端调用为 0 |
| 代码检查 | 仓库指定版本的 Black、isort、codespell，以及新增目录的 mypy 检查通过 |

发布的七个执行模块与本次 GPU 验证时的源码 SHA256 一致。验证使用 synthetic 数据和未训练参数，是功能与计时流程检查，不代表真实业务、Redis 服务或在线分配算法的性能结论。

示例输出见 [example/report.md](example/report.md)，配置／源码／模型哈希见 [example/provenance.json](example/provenance.json)。
