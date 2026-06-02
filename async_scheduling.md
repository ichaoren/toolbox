# Async Scheduling 处理流程解析

这份文档说明 `--async-scheduling` 在 vLLM V1 引擎中的实际处理流程，以及它如何影响调度器、`EngineCore` 主循环和请求生命周期。   TASK_QUEUE_ENABLE(TODO)

[TOC]

## 1. 背景

异步调度的目标不是再引入一个全新的执行线程池，而是让 CPU 侧的调度、输入准备和 GPU 侧的模型执行尽可能重叠，减少 GPU 空转间隙。

在现有实现里，这个能力主要体现在两个层面：

1. 通过 `--async-scheduling` 切换调度器实现，从普通 `Scheduler` 切到 `AsyncScheduler`。
2. 通过 `EngineCore` 中的异步执行路径，把 schedule、execute、sample、update 这些阶段拆开，用 `Future` 和 batch queue 重叠 CPU/GPU 工作。

`Model Runner V2` 的设计文档也明确把 async-first 作为设计方向之一，指出 scheduler/worker 会在 GPU 执行 step `N` 时并行为 step `N+1` 做准备。参见 [model_runner_v2.md](model_runner_v2.md)。

## 2. 开关入口与配置传递

### 2.1 CLI 参数注册

`--async-scheduling` 作为 `SchedulerConfig` 的一个布尔参数，由 [arg_utils.py](../../vllm/engine/arg_utils.py) 注册到命令行参数组中。

关键链路如下：

1. `EngineArgs` dataclass 中声明 `async_scheduling: bool | None`。
2. `EngineArgs.add_cli_args()` 在 `SchedulerConfig` 参数组内注册 `--async-scheduling`。
3. CLI 解析后，`EngineArgs.from_cli_args()` 把命令行参数回填到 `EngineArgs` 实例。

也就是说，这个开关在 CLI 层面并没有特殊分支逻辑，本质上只是一个标准配置项，先进入 `EngineArgs`，再在构造 `VllmConfig` 时下沉。

### 2.2 写入 SchedulerConfig

在 [arg_utils.py](../../vllm/engine/arg_utils.py) 的 `EngineArgs.create_engine_config()` 中，会显式构造 `SchedulerConfig`：

```python
scheduler_config = SchedulerConfig(
    ...,
    async_scheduling=self.async_scheduling,
    stream_interval=self.stream_interval,
)
```

因此，`--async-scheduling` 的配置传递链路是：

`CLI` -> `EngineArgs.async_scheduling` -> `SchedulerConfig.async_scheduling`

这里没有额外的中间适配层，也没有别名开关。后续所有行为变化都以 `SchedulerConfig.async_scheduling` 为准。

### 2.3 关键实现代码

下面这段代码就是 `EngineArgs.create_engine_config()` 中最关键的落点，说明 CLI 参数最终如何进入 `SchedulerConfig`：

```python
scheduler_config = SchedulerConfig(
    runner_type=model_config.runner_type,
    max_num_batched_tokens=self.max_num_batched_tokens,
    max_num_seqs=self.max_num_seqs,
    max_model_len=model_config.max_model_len,
    enable_chunked_prefill=self.enable_chunked_prefill,
    disable_chunked_mm_input=self.disable_chunked_mm_input,
    is_multimodal_model=model_config.is_multimodal_model,
    is_encoder_decoder=model_config.is_encoder_decoder,
    policy=self.scheduling_policy,
    scheduler_cls=self.scheduler_cls,
    max_num_partial_prefills=self.max_num_partial_prefills,
    max_long_partial_prefills=self.max_long_partial_prefills,
    long_prefill_token_threshold=self.long_prefill_token_threshold,
    disable_hybrid_kv_cache_manager=self.disable_hybrid_kv_cache_manager,
    async_scheduling=self.async_scheduling,
    stream_interval=self.stream_interval,
)
```

这段实现说明：

- `--async-scheduling` 不会在 CLI 层直接触发行为切换。
- 它先沉淀为 `SchedulerConfig.async_scheduling`。
- 真正的行为分流发生在后续调度器选择阶段。

## 3. 调度器选择路径

调度器的切换逻辑位于 [scheduler.py](../../vllm/config/scheduler.py) 的 `SchedulerConfig.get_scheduler_cls()`。

逻辑非常直接：

- 如果用户没有显式指定 `scheduler_cls`，并且 `async_scheduling` 为真，则返回 `vllm.v1.core.sched.async_scheduler.AsyncScheduler`。
- 否则返回默认的 `vllm.v1.core.sched.scheduler.Scheduler`。

对应含义是：

- `async_scheduling` 打开的主要效果，是替换调度器实现。
- 这个开关不会改变 `EngineCore` 的整体结构，也不会改写 `SchedulerInterface` 抽象接口。
- `AsyncScheduler` 不是完全独立的一套调度框架，而是继承普通 `Scheduler` 后，对少量关键方法做异步语义增强。

因此，从架构上看，这个功能的实现方式更像“在同一调度框架内切换异步语义版本”，而不是“启动一条新的并发子系统”。

### 3.1 关键实现代码

调度器切换本身只有一小段，但这是整个功能最核心的入口：

```python
def get_scheduler_cls(self) -> type["SchedulerInterface"]:
    if self.scheduler_cls is None:
        if self.async_scheduling:
            from vllm.v1.core.sched.async_scheduler import AsyncScheduler

            return AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler

        return Scheduler
```

这段代码揭示了两个事实：

- 默认情况下，`async_scheduling=True` 直接切到 `AsyncScheduler`。
- 如果用户显式传入 `scheduler_cls`，那么自定义调度器优先级更高。

## 4. EngineCore 中的主流程

### 4.1 初始化阶段

在 [core.py](../../vllm/v1/engine/core.py) 中，`EngineCore.__init__()` 会先创建执行器、初始化 KV cache，再根据 `vllm_config.scheduler_config.get_scheduler_cls()` 选择调度器类型：

```python
Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
self.scheduler = Scheduler(...)
```

初始化完成后，`EngineCore` 会记录：

- `self.async_scheduling = vllm_config.scheduler_config.async_scheduling`
- `self.batch_queue_size = self.model_executor.max_concurrent_batches`
- `self.step_fn = self.step if self.batch_queue is None else self.step_with_batch_queue`

这里要区分两个概念：

- `async_scheduling`：决定调度器语义和部分 post-step 行为。
- `batch_queue`：决定 `EngineCore` 是否使用“多 batch 排队”的执行路径。

两者有关联，但不是一回事。`batch_queue` 更偏向执行层面的异步重叠，尤其是 pipeline parallelism；`async_scheduling` 更偏向 scheduler 如何处理输出占位符、structured output 和 spec decode。

### 4.2 基础 step 路径

当没有启用 batch queue 时，`EngineCore.step()` 的主流程是：

1. `scheduler.schedule()` 产生 `SchedulerOutput`
2. `model_executor.execute_model(..., non_block=True)` 异步提交模型执行
3. 如有需要，计算 grammar bitmask
4. `future.result()` 等待执行结果
5. `scheduler.update_from_output(...)` 把模型输出写回 scheduler/request 状态

对应代码仍然是一个循环，但模型执行已经通过 `non_block=True` 先异步提交，因此 CPU 侧流程被拆成了“先发起、后等待”的结构。

关键实现如下：

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    if not self.scheduler.has_requests():
        return {}, False
    scheduler_output = self.scheduler.schedule()
    future = self.model_executor.execute_model(scheduler_output, non_block=True)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
    with (
        self.log_error_detail(scheduler_output),
        self.log_iteration_details(scheduler_output),
    ):
        model_output = future.result()
        if model_output is None:
            model_output = self.model_executor.sample_tokens(grammar_output)

    self._process_aborts_queue()
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

从这段实现能直接看出：

- `schedule()` 和 `execute_model()` 先发生；
- CPU 并不会在提交后立刻同步，而是等到真正需要结果时才 `future.result()`；
- scheduler 状态回写统一收口到 `update_from_output()`。

### 4.3 batch queue 路径

如果 `self.model_executor.max_concurrent_batches > 1`，`EngineCore` 会启用 `step_with_batch_queue()`。

这条路径的核心注释很明确：batch queue 允许异步地 schedule 和 execute batch，并且是 pipeline parallelism 消除 bubble 的必要条件。

它的处理顺序是：

1. 尝试在 batch queue 未满时继续 schedule 新 batch。
2. 立即把 `execute_model(..., non_block=True)` 返回的 future 放入队列。
3. 如果当前还能继续填充队列，则优先继续排队，而不是马上等待最早结果。
4. 当队列已满或没有更多请求可调度时，再阻塞等待最早一个 batch 完成。
5. 完成后调用 `scheduler.update_from_output(...)` 更新状态。

这意味着 `EngineCore` 层面已经具备“上一批 GPU 正在跑，CPU 继续为下一批做调度/采样准备”的能力。`AsyncScheduler` 的作用，则是在 scheduler/request 状态层面把这种重叠变得可维护。

关键实现如下：

```python
if self.scheduler.has_requests():
    scheduler_output = self.scheduler.schedule()
    with self.log_error_detail(scheduler_output):
        exec_future = self.model_executor.execute_model(
            scheduler_output, non_block=True
        )
    ...
    if not scheduler_output.pending_structured_output_tokens:
        grammar_output = self.scheduler.get_grammar_bitmask(
            scheduler_output
        )
        future = self.model_executor.sample_tokens(
            grammar_output, non_block=True
        )
    else:
        deferred_scheduler_output = scheduler_output

    if not deferred_scheduler_output:
        batch_queue.appendleft((future, scheduler_output, exec_future))
        if (
            model_executed
            and len(batch_queue) < self.batch_queue_size
            and not batch_queue[-1][0].done()
        ):
            return None, True
```

这里最关键的点有两个：

- `pending_structured_output_tokens` 会决定采样是立即进行还是延迟。
- batch queue 未满时，`EngineCore` 倾向于继续灌队列，而不是立刻等待前一个 batch 结果。

### 4.4 post_step 的差异

`EngineCore.post_step()` 中有一段与异步调度直接相关的逻辑：

```python
if not self.async_scheduling and self.use_spec_decode and model_executed:
    draft_token_ids = self.model_executor.take_draft_token_ids()
    ...
```

也就是说：

- 非 async 模式下，draft token ids 由 `EngineCore.post_step()` 在主进程侧拿回并写回 scheduler。
- async 模式下，代码注释明确说明“拿不到提前的 draft token ids”，因此改为在 worker 侧更新，不再在这里处理。

这是一个非常关键的分界点：异步调度不仅改变了 scheduler 本身，还改变了 spec decode 的 token 回填位置。

## 5. AsyncScheduler 与普通 Scheduler 的关系

`AsyncScheduler` 定义在 [async_scheduler.py](../../vllm/v1/core/sched/async_scheduler.py)，实现非常克制：

- 直接继承 `Scheduler`
- 只覆写少量关键方法
- 复用 `SchedulerInterface` 和绝大多数基线逻辑

这说明异步调度并不是推翻普通调度器，而是在普通调度器已经具备的 persistent-state / incremental-update 框架上，补上异步场景下最难处理的状态一致性问题。

核心实现如下：

```python
class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
```

从类定义就能看出：

- 它完全复用 `Scheduler` 主体逻辑；
- 新增的状态非常少，主要就是 speculative decoding 的占位符数组；
- 设计重点是“少改基类、只改异步关键路径”。

## 6. AsyncScheduler 的关键差异

### 6.1 `_update_after_schedule()`

`AsyncScheduler._update_after_schedule()` 在调用父类实现之后，会继续处理输出占位符和 speculative decoding 占位：

```python
for req_id in scheduler_output.num_scheduled_tokens:
    request = self.requests[req_id]
    if request.is_prefill_chunk:
        continue

    scheduler_output.pending_structured_output_tokens |= (
        request.use_structured_output and request.num_output_placeholders > 0
    )
    cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
    request.num_output_placeholders += 1 + cur_num_spec_tokens
    request.spec_token_ids = self._spec_token_placeholders
```

这段逻辑体现了 async 调度的核心思想：在真正的输出 token 还没有完全回传前，先在 request 状态里放入“未来将要到账”的占位符。

这些占位符的作用有三个：

1. 让 scheduler 在下一次 schedule 时知道这个请求已经“欠着一些待兑现输出”。
2. 让 structured output 路径知道当前 grammar bitmask 可能还不能立刻计算。
3. 让 speculative decoding 在 worker 真正回填 draft token 之前，先保留 token 数量信息。

完整实现如下：

```python
def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
    super()._update_after_schedule(scheduler_output)
    spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
    for req_id in scheduler_output.num_scheduled_tokens:
        request = self.requests[req_id]
        if request.is_prefill_chunk:
            continue

        scheduler_output.pending_structured_output_tokens |= (
            request.use_structured_output and request.num_output_placeholders > 0
        )
        cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
        request.num_output_placeholders += 1 + cur_num_spec_tokens
        request.spec_token_ids = self._spec_token_placeholders
```

### 6.2 `_update_request_with_output()`

异步调度器对输出回写也做了覆盖：

```python
if request.discard_latest_async_tokens:
    request.discard_latest_async_tokens = False
    return [], False
```

如果请求被强制前缀缓存重置打断，那么最新一个异步 token 需要被丢弃，避免重复输出。

随后它会调用父类逻辑完成正常 token 追加与 stop 检查，然后做两件异步专属清理：

1. `request.num_output_placeholders -= len(new_token_ids)`
2. 对仍处于 `RUNNING` 状态的请求，按 `num_computed_tokens - num_output_placeholders` 把 KV block 写入缓存

也就是说，在 async 模式下，请求的“已经计算过多少 token”和“已经真正确认输出了多少 token”不再完全等价，中间要靠 `num_output_placeholders` 把这段时间差显式表示出来。

完整实现如下：

```python
def _update_request_with_output(
    self, request: Request, new_token_ids: list[int]
) -> tuple[list[int], bool]:
    if request.discard_latest_async_tokens:
        request.discard_latest_async_tokens = False
        return [], False

    status_before_update = request.status
    new_token_ids, stopped = super()._update_request_with_output(
        request, new_token_ids
    )

    request.num_output_placeholders -= len(new_token_ids)
    assert request.num_output_placeholders >= 0

    if status_before_update == RequestStatus.RUNNING:
        self.kv_cache_manager.cache_blocks(
            request, request.num_computed_tokens - request.num_output_placeholders
        )
    return new_token_ids, stopped
```

## 7. 普通 Scheduler 中已经存在的异步兼容逻辑

虽然 `AsyncScheduler` 只改了少量方法，但普通 [scheduler.py](../../vllm/v1/core/sched/scheduler.py) 本身已经包含了不少 async-aware 分支。

### 7.1 运行队列中的额外 step 避免

在 `schedule()` 的 RUNNING 请求遍历里，存在如下判断：

```python
if (
    request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens
):
    continue
```

这段注释写得很清楚：在 async scheduling 下，如果可以确定前一步已经让请求达到 `max_tokens`，就不要再额外调度一步。

这也是为什么 `num_output_placeholders` 不是一个附属字段，而是下一轮调度决策的直接输入。

对应实现如下：

```python
if (
    request.num_output_placeholders > 0
    and request.num_computed_tokens + 2 - request.num_output_placeholders
    >= request.num_prompt_tokens + request.max_tokens
):
    req_index += 1
    continue
```

### 7.2 RUNNING 请求的 token 预算计算

普通调度器给 RUNNING 请求计算 `num_new_tokens` 时，不是简单地用：

`request.num_tokens_with_spec - request.num_computed_tokens`

而是：

`request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens`

这说明 scheduler 已经把“未来待兑现的输出 token”也纳入了调度账本。

对应实现如下：

```python
num_new_tokens = (
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
)
```

### 7.3 CachedRequestData 的输出 token 统计

在 `_make_cached_request_data()` 中，`num_output_tokens` 记录的不是单纯的 `req.num_output_tokens`，而是：

```python
req.num_output_tokens + req.num_output_placeholders
```

因此，传给 model runner 的 cached request 视图，也会显式感知这些占位符。这是异步场景下维持 persistent batch 正确性的关键之一。

对应实现如下：

```python
num_output_tokens.append(
    req.num_output_tokens + req.num_output_placeholders
)
```

### 7.4 Structured output 的延迟 grammar 计算

普通调度器的 `get_grammar_bitmask()` 只有在 `scheduler_output.has_structured_output_requests` 为真时才会继续构造 bitmask。

而 `AsyncScheduler._update_after_schedule()` 还会设置：

- `scheduler_output.pending_structured_output_tokens`

这个字段会在 `EngineCore.step_with_batch_queue()` 中决定是否立即 `sample_tokens()`，还是先推迟，等拿到上一轮输出之后再计算 grammar bitmask。

所以在异步模式下，structured output 的 grammar 计算可能被延后，这不是额外 feature，而是为了解决“CPU 还没有拿到足够 token，无法安全生成 grammar bitmask”的时序问题。

## 8. 请求生命周期视角

下面按一次请求从进入到完成的顺序，串起异步调度的关键状态变化。

### 8.1 请求进入

`EngineCore.add_request()` 最终会调用 `scheduler.add_request()`；调度器内部将请求放入 waiting 队列，并登记到 `self.requests`。

请求对象定义在 [request.py](../../vllm/v1/request.py)，其中两个字段专门服务异步调度：

- `num_output_placeholders = 0`
- `discard_latest_async_tokens = False`

### 8.2 第一次被 schedule

`scheduler.schedule()` 会把请求从 waiting 提升到 running，分配 KV blocks，构造 `SchedulerOutput`。

`SchedulerOutput` 定义在 [output.py](../../vllm/v1/core/sched/output.py)，与异步调度最相关的字段包括：

- `num_scheduled_tokens`
- `scheduled_spec_decode_tokens`
- `has_structured_output_requests`
- `pending_structured_output_tokens`
- `num_invalid_spec_tokens`

### 8.3 schedule 之后立即更新逻辑状态

`Scheduler._update_after_schedule()` 会先把 `request.num_computed_tokens` 往前推进，因为下一轮调度要基于“已经计划出去”的 token 数继续计算。

如果使用的是 `AsyncScheduler`，这一步还会额外增加 `num_output_placeholders`，把未来将回传的 output token 和 spec token 先占位。

### 8.4 GPU 执行与输出回传

`EngineCore` 通过 `execute_model(..., non_block=True)` 把工作先提交给执行器，然后在合适的时刻等待 `future.result()`。

如果启用了 batch queue，CPU 甚至可能在等待当前 batch 完成前，继续为下一批做 schedule。

### 8.5 scheduler 用输出修正状态

执行完成后，`scheduler.update_from_output(...)` 会：

1. 读取 `sampled_token_ids`
2. 结合 `scheduled_spec_decode_tokens` 计算接受/拒绝的 spec token 数
3. 必要时回退 `request.num_computed_tokens`
4. 在 async 场景下同步回退 `request.num_output_placeholders`
5. 调用 `_update_request_with_output()` 追加真实输出 token
6. 做 stop 检查、结构化输出推进、资源回收与结果封装

其中一个关键分支是：

```python
if request.num_output_placeholders > 0:
    request.num_output_placeholders -= num_rejected
```

这说明 placeholder 不只是“先加后减”的计数器，它还参与了 speculative decoding 的接受率修正。

### 8.6 请求完成或被打断

请求完成时，`finish_requests()` / `_free_request()` / `_free_blocks()` 会释放 scheduler 和 KV cache 中的状态。

如果发生 `reset_prefix_cache(reset_running_requests=True)`，所有 running 请求会被强制 preempt，并设置：

```python
request.num_output_placeholders = 0
request.discard_latest_async_tokens = True
```

原因是异步调度中最新一个 token 可能已经在“路上”，如果不丢弃，恢复后会出现冗余重复 token。

## 9. 与 speculative decoding、structured output 的耦合

异步调度的复杂度，主要来自它和两个功能的叠加：

### 9.1 speculative decoding

在普通路径下，draft token ids 可以在 `post_step()` 里由 `EngineCore` 统一取回。

但在异步路径下，代码明确改为在 worker 侧更新；`AsyncScheduler` 这里只先放 `-1` 占位，后续再由 worker 的真实结果回填。

这会影响：

- `request.spec_token_ids`
- `scheduled_spec_decode_tokens`
- `num_invalid_spec_tokens`
- `num_output_placeholders`

### 9.2 structured output

structured output 依赖 grammar bitmask，而 grammar bitmask 又依赖当前请求到底已经确认了哪些 token。

异步调度下，CPU 侧在某个时间点可能只有 placeholder，还没有最终 token，因此：

- 有时可以立即 `sample_tokens()`
- 有时必须先 defer，等前一轮结果回来再算 grammar bitmask

`EngineCore.step_with_batch_queue()` 中的 `deferred_scheduler_output` 分支，就是为这个问题准备的。

关键实现如下：

```python
if not scheduler_output.pending_structured_output_tokens:
    grammar_output = self.scheduler.get_grammar_bitmask(
        scheduler_output
    )
    future = self.model_executor.sample_tokens(
        grammar_output, non_block=True
    )
else:
    deferred_scheduler_output = scheduler_output
```

以及延迟处理分支：

```python
if deferred_scheduler_output:
    if self.use_spec_decode:
        draft_token_ids = self.model_executor.take_draft_token_ids()
        assert draft_token_ids is not None
        self.scheduler.update_draft_token_ids_in_output(
            draft_token_ids, deferred_scheduler_output
        )
    grammar_output = self.scheduler.get_grammar_bitmask(
        deferred_scheduler_output
    )
    future = self.model_executor.sample_tokens(grammar_output, non_block=True)
```

## 10. 关键约束与边界

从当前代码实现看，可以总结出几个重要边界：

1. `--async-scheduling` 的实现重点是调度器切换和状态建模，不是新增一个独立异步服务线程。
2. `AsyncScheduler` 只覆盖少量方法，说明异步路径仍然依赖普通 `Scheduler` 的主体流程和数据结构。
3. `num_output_placeholders` 是 async 语义的核心状态，它影响 schedule、cached request 构造、spec token 修正和 stop 边界判断。
4. `batch_queue` 与 `async_scheduling` 不是同一个开关，但两者共同决定 CPU/GPU 是否能更深度重叠。
5. 当前异步路径对 structured output 和 speculative decoding 的处理明显更复杂，这也是它主要的代码增量来源。
6. 这套分析完全基于当前仓库实现；文档不推断未在代码中出现的上游优化或性能结论。

## 11. 结论

`--async-scheduling` 在 vLLM V1 中的核心作用，可以概括为三句话：

1. 它先在配置层面把默认调度器切换为 `AsyncScheduler`。
2. 它通过 `num_output_placeholders` 等状态，把“GPU 正在执行但结果尚未完全回到 CPU”的时间窗口显式建模出来。
3. 它再配合 `EngineCore` 的非阻塞执行与 batch queue，让调度、采样、structured output 校验和 GPU 执行尽可能重叠。

如果要继续深入读源码，最值得优先看的文件是：

- [arg_utils.py](../../vllm/engine/arg_utils.py)
- [scheduler.py](../../vllm/config/scheduler.py)
- [core.py](../../vllm/v1/engine/core.py)
- [scheduler.py](../../vllm/v1/core/sched/scheduler.py)
- [async_scheduler.py](../../vllm/v1/core/sched/async_scheduler.py)
- [request.py](../../vllm/v1/request.py)
- [output.py](../../vllm/v1/core/sched/output.py)
