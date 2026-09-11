# Recsys LayerReady Stream Scheduler

分支：`recsys-layer-ready-scheduler`（worktree 基于
[PR 428](https://github.com/NVIDIA/recsys-examples/pull/428) `e31c5ef`）。

P0 已接 CUDA IPC layer-ready event：FlexKV worker 在每层 H2D 后
`cudaEventRecord`（`cudaEventInterprocess`），经 UDS 把 IPC handle 交给 Recsys；
`HostKVTaskHandle.stream_wait_layer` 优先 `gpu_ready_events` 上的
`cudaStreamWaitEvent`，eventfd 仍作回退。`ssd_read` / `h2d_layer` 按 span
拆开的原语尚未接上。

用 FlexKV 暴露的完成信号 + Recsys 侧 CUDA stream 组织 onboard 调度，
取代把 naive / layerwise / 切盘 / GDS / prefetch 分叉堆进
`third_party/FlexKV/csrc/layerwise.cpp`。

产品名只有 **naive** 和 **layerwise**。切盘、GDS、prefetch 是同一套
`LayerReadyPlan` 的字段，不是第三种模式。就绪粒度永远是 **original
layer**（8 层模型 wait 的是 layer 0…7）。

---

## 1. 动机

今天 FlexKV 的 layerwise 路径把三件事绑在一次 `client.launch` 里：

1. SSD 怎么切（`FLEXKV_PIPELINE_SSD_LAYER_BATCH`、均分 / 不均分）
2. H2D / GDS 什么时候提交
3. 完成后写 **eventfd**，Recsys `os.read` 再打 HSTU

Recsys 几乎只做：

- naive：`onboard_wait` 整包，再 8 层连算
- layerwise：每层 replay / eager kernel 前 `wait_layer(i)`（CPU 阻塞读 fd）

后果：

- 每加一种重叠（three-pipeline、N-layerwise、GDS notify）都要改 FlexKV C++
- Recsys 的 `stream_wait_layer` **名不副实**：FlexKV 路径并没有
  `cudaStreamWaitEvent`，只是 CPU `os.read`
- Native host KV 已经是 stream + CUDA event，FlexKV 却走了另一套 IPC fd

目标不是「零改 FlexKV」，而是把 FlexKV **收成传输原语**，把拍子交给 Recsys。

---

## 2. 目标与非目标

### 2.1 目标

| ID | 内容 |
| --- | --- |
| G1 | Recsys 拥有 `LayerReadyPlan` 和唯一的 wait 点；HSTU 只等 **GPU-ready** |
| G2 | FlexKV 不再 for-loop 里 fill / prefetch / notify；不再靠 env 开关调度 |
| G3 | 同一套循环覆盖 naive、普通 D2H+HSTU layerwise、SSD even N、GDS、prefetch |
| G4 | GPU 侧用 CUDA event + `cudaStreamWaitEvent`，与 Native host KV 同构 |
| G5 | io_uring / cuFile 仍在 FlexKV；Python 不打盘、不自己 `cudaMemcpy` KV |

### 2.2 非目标（本设计不做）

- 把 io_uring for-loop 搬进 Python
- 按 span 多次跨进程 `client.launch`（RPC 太贵）
- uneven 切盘、GDS 上的 prefetch、SWA sidecar、GET 内 `prefetch_depth > 1`
- 改 KV 布局或 block 大小
- 改「一层一通知」为 span 级 wait（HSTU 仍按 original layer）

---

## 3. 现状

### 3.1 Native host KV（要对齐的模型）

`native_host_kvcache_manager_impl.cpp`：

```
onload_stream:  H2D(layer i) → record internal_onload_event[i]
scatter_stream: wait(internal_onload_event[i]) → scatter → record compl_event[i]
compute stream: wait_layer(i) = CPU 等到 host_complete[i] 后
                cudaStreamWaitEvent(current, compl_event[i])
```

真正放行 HSTU 的是 **CUDA event**。CPU 条件变量只保证 event 已经 record。

### 3.2 FlexKV layerwise（要拆掉的调度）

```
FlexKV 私有 copy stream:
  SSD span (io_uring) → H2D(layer) → cudaEventRecord
  HOSTFUNC / polling → write(eventfd[layer])

Recsys:
  wait_layer(i) = os.read(eventfd[i], 8)     # CPU 阻塞
  然后才 replay HSTU(i) 或 eager kernel
```

内部已经有 `cudaEventRecord`，但没有交给 Recsys 做 `stream.wait_event`。
eventfd 还要 Unix socket + SCM_RIGHTS 把 fd 交给 FlexKV worker
（`flexkv_layerwise.py`）。

`HostKVTaskHandle.stream_wait_layer` 对 FlexKV 只是转调 `handle.wait_layer`。

### 3.3 Graph 路径

`hstu_block_inference.py`：每层 graph replay **之前** CPU `wait_layer(idx-1)`。
event 没进图，所以即使用 CUDA event，第一版也可以保持「replay 前同步」。

---

## 4. 完成信号：必须分两层

SSD 完成发生在 CPU（io_uring CQ），**不能**直接变成 `cudaEvent`。
GPU 上的数据就绪发生在 H2D 或 GDS 之后，这时才可以 `cudaEventRecord`。

| 完成点 | 信号 | 谁 wait | 为什么 |
| --- | --- | --- | --- |
| SSD→CPU 某个 span 读完 | CPU future / callback | Recsys scheduler | io_uring 在 host |
| 层 i 的字节已在 GPU | `cudaEvent`（`DisableTiming`） | compute stream | 可 `WaitEvent`，不必 CPU 阻塞 HSTU |
| CUDA Graph | 图内 `WaitEvent`，或 replay 前 `event.synchronize()` | 同现网 | 与今天 graph 前 wait 等价 |

prefetch 只铺 CPU：完成信号是同一套 SSD future，GET 时已 commit 的层跳过
`ssd_read`，直接 `h2d_layer`。

GDS 没有 H2D：span 的 GPU-ready event 绑在 cuFile 完成之后。

---

## 5. 职责划分

```
┌─────────────────────────────────────────────────────────────┐
│ Recsys  LayerReadyScheduler                                 │
│  · 持有 LayerReadyPlan                                      │
│  · 拥有 copy_stream / compute_stream                        │
│  · 提交下一 span、搭拍、wait GPU-ready、打 HSTU(i)          │
│  · naive：等最后一个 layer event 再 8 层连算                │
└───────────────┬─────────────────────────────┬───────────────┘
                │ ssd_read / h2d / gds        │ wait_event
                ▼                             ▼
┌───────────────────────────────┐    ┌────────────────────────┐
│ FlexKV 传输原语               │    │ HSTU (eager / graph)   │
│  ssd_read(span) → CpuFuture   │    │ 只看见 layer 已在 GPU  │
│  h2d_layer(i, stream) → Event │    └────────────────────────┘
│  gds_read(span, stream)→Event │
│  不做调度、不写 eventfd       │
└───────────────────────────────┘
```

| 谁 | 做 | 不做 |
| --- | --- | --- |
| FlexKV | 盘、页表、CPU pin buffer、H2D/GDS 实际提交；返回 future / event | THREE_PIPELINE、LAYER_BATCH、even/uneven for-loop、HOSTFUNC 写 fd |
| Recsys scheduler | 按 plan 提交、1-deep 预读下一 span、stream 搭拍 | io_uring、cuFile、自己拷 KV |
| HSTU | `wait` 当前层 GPU-ready 后算该层 | 知道 span / GDS / prefetch |

---

## 6. LayerReadyPlan

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class Span:
    start: int   # inclusive original layer
    end: int     # exclusive

@dataclass(frozen=True)
class LayerReadyPlan:
    transport: Literal["uring_ssd", "gds"]
    ready: Literal["naive", "layerwise"]
    spans: tuple[Span, ...]          # even N，或单 span [0, L)
    prefetch: Literal["off", "whole", "layer"]
    prefetch_depth: int = 1          # GET 内 1-deep；>1 本阶段不做
```

字段语义：

| 字段 | 取值 | 含义 |
| --- | --- | --- |
| `transport` | `uring_ssd` | SSD→CPU staging→H2D |
| | `gds` | cuFile 直达 GPU，无 H2D、无 CPU prefetch |
| `ready` | `naive` | 计算侧等整次 GET（最后一个 GPU-ready event）再连算 |
| | `layerwise` | 每层 `wait` GPU-ready[i] 再 HSTU(i) |
| `spans` | `[0,L)` | 盘 / GDS 一次读完（普通 layerwise 或 naive） |
| | even N=1/2/4 | 均分切盘；第一段长度 = N |
| `prefetch` | `off` | GET 时才读 |
| | `whole` | 空闲整包 SSD→CPU commit；GET 变 CPU-hit |
| | `layer` | 空闲按层或按 span commit；GET 对已在 CPU 的层只 H2D |

`prefetch` **只针对 SSD→CPU→GPU**。GDS 路径本阶段 `prefetch=off`。

工厂（示意）：

```python
def even_spans(num_layers: int, n: int) -> tuple[Span, ...]:
    return tuple(Span(i, min(i + n, num_layers)) for i in range(0, num_layers, n))

NAIVE          = LayerReadyPlan("uring_ssd", "naive",     (Span(0, 8),), "off")
LAYERWISE_H2D  = LayerReadyPlan("uring_ssd", "layerwise", (Span(0, 8),), "off")
SSD_EVEN_N     = lambda n: LayerReadyPlan("uring_ssd", "layerwise", even_spans(8, n), "off")
NAIVE_GDS      = LayerReadyPlan("gds", "naive", (Span(0, 8),), "off")
GDS_EVEN_N     = lambda n: LayerReadyPlan("gds", "layerwise", even_spans(8, n), "off")
NAIVE_PREFETCH = LayerReadyPlan("uring_ssd", "naive", (Span(0, 8),), "whole")
LW_PREFETCH    = lambda n: LayerReadyPlan("uring_ssd", "layerwise", even_spans(8, n), "layer")
```

CLI 用 `--chunk-policy` / `--transport` / `--prefetch` 替换
`FLEXKV_PIPELINE_SSD_LAYER_BATCH`、`FLEXKV_PIPELINE_SSD_EVEN_SPLIT`、
`THREE_PIPELINE` 三个 env。默认切盘 N 仍为 **1**（与现网默认一致）。

---

## 7. FlexKV 原语（建议 API）

进程内（embedded KVManager / 同进程 worker）多次 submit 可接受。
**不要**做成一 span 一次跨进程 RPC。

```python
class CpuSpanFuture:
    """SSD→CPU 完成。wait() 阻塞当前 CPU 线程；done() 轮询。"""
    def done(self) -> bool: ...
    def wait(self) -> None: ...

class GpuReadyEvent:
    """层或 span 在 GPU 上就绪。底层是 cudaEvent_t（DisableTiming）。"""
    def wait_stream(self, stream: torch.cuda.Stream) -> None:
        stream.wait_event(self._event)
    def synchronize(self) -> None:
        self._event.synchronize()

class FlexKVTransport:
    def ssd_read(self, task, span: Span) -> CpuSpanFuture:
        """io_uring 读 span 覆盖的 original layers 到 CPU pin buffer。"""

    def h2d_layer(self, task, layer: int, stream: torch.cuda.Stream) -> GpuReadyEvent:
        """在给定 stream 上提交该层 H2D（含必要 scatter），record event 后返回。
        调用方必须保证覆盖该层的 ssd_read 已 wait 完。"""

    def gds_read(self, task, span: Span, stream: torch.cuda.Stream) -> GpuReadyEvent:
        """cuFile 读 span 到 GPU。返回的 event 在 span 内每一层都可用，
        或按层拆成 L 个 event（实现可选；对外 wait 粒度仍是 layer）。"""

    def cpu_layer_ready(self, task, layer: int) -> bool:
        """prefetch commit 后：该层是否已在 CPU。GET 时跳过 ssd_read。"""
```

约束：

1. `h2d_layer` / `gds_read` **必须**吃 Recsys 传入的 `stream`，不要再开
   FlexKV 私有 copy stream 让 Recsys 看不见。
2. 返回的 `GpuReadyEvent` 在 task 生命周期内有效；`onboard_wait` / 析构时销毁。
3. `ssd_read` 对同一 task 同一 span 幂等；已在 CPU 则立即 done。
4. Python 不拿到 raw KV pointer 自己拷。

### 7.1 谁 record event

推荐 **FlexKV 在用户 stream 上拷完后 record，并返回 event**
（GDS 只能这样，因为 Recsys 没有 enqueue cuFile）。

等价写法：若 `h2d_layer` 保证只往 `stream` 上提交、返回时 enqueue 已结束，
Recsys 也可以自己 `event.record(stream)`。两种都合法；对外接口仍返回
`GpuReadyEvent`，避免 Graph / 多 GPU 时调用方漏 record。

### 7.2 层 event 与 span event

HSTU 的 wait 粒度是 layer。实现上：

- `uring_ssd`：一层一次 H2D → 一层一个 event（与 Native `compl_event[i]` 相同）
- `gds` + layerwise：一个 GDS span 完成后，对该 span 内每一层 **record 或复用**
  同一个 event；`wait_layer(i)` 等覆盖 i 的那个 event
- naive：scheduler 只 wait 最后一个 layer / 整包 event

---

## 8. Recsys Scheduler

### 8.1 Stream 拓扑

| Stream | 用途 | 等待 |
| --- | --- | --- |
| `copy_stream` | H2D 或 GDS | CPU 确认 span 在 pin buffer（GDS 则无此步） |
| `compute_stream` | HSTU eager / graph replay | `GpuReadyEvent[i]` |

与 Native 的 onload/scatter 双 stream 不同：scatter 若必须留在 FlexKV 内部，
由 `h2d_layer` 在 `copy_stream` 上串完 memcpy+scatter 再 record **一个**
对外 event。Recsys 只看见「层 i 在 GPU」。

### 8.2 layerwise + uring_ssd（含 even N）

```
cpu[0] = ssd_read(spans[0]);  cpu[0].wait()
inflight = ssd_read(spans[1]) if len(spans) > 1 else None     # 1-deep

for i in range(L):
    wait_cpu_span_covering(i)                 # 必要时 wait inflight 并提交 spans[k+1]
    ev[i] = h2d_layer(i, stream=copy_stream)
    compute_stream.wait_event(ev[i])
    hstu_layer(i, stream=compute_stream)
```

`wait_cpu_span_covering(i)`：

- 若 layer i 已被 prefetch commit → 直接返回
- 若当前 inflight span 覆盖 i 且尚未 done → `inflight.wait()`，再
  `ssd_read(next_span)`（depth=1）
- 普通 layerwise `spans=[0,L)`：循环前盘已一次读完，循环内不再 `ssd_read`

### 8.3 naive

同一套原语，外圈不同：

```
ssd_read([0,L)).wait()
for i in range(L):
    ev[i] = h2d_layer(i, copy_stream)
compute_stream.wait_event(ev[L-1])     # 或单独的 whole-cache event
for i in range(L):
    hstu_layer(i, compute_stream)      # 不再 per-layer wait
```

删除 `inference_dense_module` 里 `not is_layerwise` 的整包 `onboard_wait`
特殊分支：naive 变成「scheduler 在第一层计算前 wait 最后一个 event」。

### 8.4 GDS

```
# naive GDS
ev = gds_read([0,L), copy_stream)
compute_stream.wait_event(ev)
for i in range(L): hstu_layer(i)

# GDS + layerwise even N
inflight = gds_read(spans[0], copy_stream)
for i in range(L):
    if i == spans[k].start and k+1 < n_spans:
        next_ev = gds_read(spans[k+1], copy_stream)   # 与当前 HSTU 重叠
    compute_stream.wait_event(event_covering(i))
    hstu_layer(i)
```

无 H2D。prefetch 字段忽略。

### 8.5 prefetch（仅 uring_ssd）

空闲（上一请求 HSTU 完成后、下一请求 GET 前）：

- `whole`：`ssd_read([0,L))` 并 commit CPU
- `layer`：按与 GET 相同的 `spans` 逐段 `ssd_read` + commit

GET：`cpu_layer_ready(i)` 为真则跳过该层所属 span 的盘读，只 `h2d_layer`。

---

## 9. 场景对照

| 场景 | transport | ready | spans | prefetch | 重叠 |
| --- | --- | --- | --- | --- | --- |
| naive | uring_ssd | naive | `[0,8)` | off | GET 返回后连算 |
| 普通 layerwise | uring_ssd | layerwise | `[0,8)` | off | H2D(i) ∥ HSTU(i−1)；盘先做完 |
| SSD even N=1/2/4 | uring_ssd | layerwise | 均分 | off | SSD(下一 span) ∥ H2D(i) ∥ HSTU(i−1) |
| naive GDS | gds | naive | `[0,8)` | off | GDS 完再连算 |
| GDS + layerwise | gds | layerwise | even N（默认 4） | off | GDS(下一 span) ∥ HSTU |
| naive prefetch | uring_ssd | naive | `[0,8)` | whole | 空闲整包 SSD→CPU；GET 只 H2D |
| layerwise + prefetch | uring_ssd | layerwise | 与 GET 相同 | layer | 已 commit 的层 GET 只 H2D |

auto 建议（均分、先 SSD）：

- 短 seq（2K/4K）：layerwise even N=4，可选 `prefetch=layer`
- 16K、盘占 e2e 高：even N=2 或 4
- 有 GDS、压 e2e：`gds` + layerwise even N=4
- 只要突发带宽：naive / naive GDS
- 请求间隔长：同一 ready/spans，打开 prefetch

---

## 10. CUDA Graph

两档，可分阶段：

**V1（与现网同构）**  
每层 replay 前 CPU：`ev[i].synchronize()` 或短 `wait_layer`。
把 eventfd 换成 CUDA event，图本身不变。eager 路径改为
`compute_stream.wait_event`（真正的 stream wait）。

**V2**  
capture 时把 `WaitEvent(ev[i]) + HSTU(i)` 收进同一 graph。
要求 `ev[i]` 在 capture 外 record（或使用 graph-external event）。
H2D 仍在 `copy_stream` 上、在 capture 之外提交——与「拷贝 stream ∥ 计算
graph」的常见拆分一致。

不要把 SSD `wait()` 放进 graph。

---

## 11. 进程模型

| 模式 | span 级原语 | 建议 |
| --- | --- | --- |
| embedded，worker 同进程或可共享 CUDA context | 多次 `ssd_read` / `h2d_layer` | **默认落地路径** |
| external server | 跨进程 CUDA event 需 `cudaIpcGetEventHandle`；每 span RPC 贵 | 一次 RPC 带上整个 plan，worker 内按原语循环，**只把 layer event 句柄数组送回 Recsys** |

现网 eventfd 已经是「Recsys 建 fd、SCM_RIGHTS 交给 worker」。CUDA event
IPC 可以走同一条控制面，但 scheduler 逻辑仍在 Recsys；若 RPC 只能一次
launch，则 worker 执行的是 Recsys 序列化后的 plan，而不是 C++ 里的 bool
分叉。优先 embedded，避免一上来做 event IPC。

---

## 12. Recsys 代码落点

| 模块 | 变化 |
| --- | --- |
| 新 `layer_ready_scheduler.py` | Plan、span 工厂、8.2–8.5 循环 |
| `flex_kvcache_manager.py` | `onboard_launch(plan)`；handle 持有 `GpuReadyEvent[]` 而非 eventfd |
| `_FlexKVOnloadHandle.wait_layer` | `compute_stream.wait_event(ev[i])`；保留可选 `synchronize()` 给 graph V1 |
| `host_kvstorage_manager.py` | `stream_wait_layer` 名实相符 |
| `inference_dense_module.py` | 删除 `not is_layerwise` 整包 wait；naive 由 scheduler 在首层前 wait |
| `hstu_block_inference.py` | 仍 `wait_layer(idx-1)`，底层换成 event |
| `paged_hstu_infer_layer.py` | `stream_wait_layer` 走 CUDA wait |
| CLI / `run_pr428_ssd_hit_case.sh` | `--chunk-policy` 替换三个 FlexKV env |

FlexKV：

- `layerwise.cpp` 提供 `ssd_read` / `h2d_layer` / `gds_read` 绑定
- 删除 HOSTFUNC→eventfd 作为主路径（可留兼容开关一个版本）
- 调度 env 不再被 Recsys 实验脚本依赖

---

## 13. 兼容与迁移

1. **行为兼容**：`ready=layerwise, spans=[0,L), prefetch=off` ≡ 今天的
   「盘一次读完 + 逐层 H2D + wait_layer」。
2. **eventfd 过渡**：`FLEXKV_LAYERWISE_NOTIFY=eventfd|cuda_event`。
   cuda_event 为新默认；eventfd 仅用于对照实验。
3. **默认 N=1**：even 切盘默认仍一层一盘，避免静默改变现网 e2e。
4. 正确性：与现有 `test_flexkv.py` layerwise、Native `test_native.py`
   同一套 per-layer wait 断言。

---

## 14. 风险

| 风险 | 处理 |
| --- | --- |
| Python 循环提交 H2D 的开销 | 一层一次 enqueue 与今天 C++ 一层一次相同量级；不要一层多次 Python launch |
| `copy_stream` 与 FlexKV 内部 stream 混用 | API 强制外部传入 stream；FlexKV 内部只借用 |
| Graph capture 与跨 stream event | 先 V1 CPU sync，再 V2 external event |
| GDS 完成如何绑到用户 stream | FlexKV 在该 stream 上 record；或 cuFile 完后 host 回调再 record |
| worker 进程 CUDA context 不同 | 先 embedded；IPC 作为第二阶段 |
| 过早 wait CPU span 把 HSTU 堵在 host | `wait_cpu` 只在该层 H2D 前；HSTU 只 wait GPU event |

---

## 15. 落地阶段

| 阶段 | 内容 | 完成标准 |
| --- | --- | --- |
| P0 Recsys 骨架（本分支） | `LayerReadyPlan`、handle 上 `gpu_ready_events`、`stream_wait_layer` 优先 CUDA event 否则 eventfd；naive 走 `needs_whole_onboard_wait()` | 纯 Python 单测通过；现网 launch 行为不变 |
| P0 FlexKV event（本分支） | worker 创建 interprocess `cudaEvent`，H2D 后 record，UDS 导出 IPC handle；Recsys `ImportedCudaEvent.wait_stream` | layerwise e2e 与 eventfd 路径数值一致 |
| P1 | `ssd_read(span)` + Recsys 1-deep 循环；even N=1/2/4 | 对齐 three-pipeline / N-layerwise 实验 |
| P2 | `gds_read(span, stream)`；naive 走同一 scheduler | 删掉 dense_module 整包 `client.wait` |
| P3 | prefetch whole/layer | 空闲 commit；GET 跳过已 ready 层 |
| P4 | Graph V2；external 模式 event IPC（可选） |  |

P0 已经能「摆脱在 FlexKV 里加调度分叉」：后面每种场景只改 Recsys plan。

---

## 16. 一句话

FlexKV 变成 `ssd_read` / `h2d_layer` / `gds_read` 三个带完成信号的原语；
Recsys 用 `copy_stream` 与 `compute_stream` 上的 CUDA event 搭拍。
SSD 完成留在 CPU future，GPU 就绪才是 event。调度从 `layerwise.cpp`
搬到 Recsys，传输留在 FlexKV。
