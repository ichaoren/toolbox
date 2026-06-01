## 1. 问题现象

服务运行过程中在 NPU 上发生 AICore 异常，主进程随后触发 shutdown。应用日志中的关键报错如下：

```text
RuntimeError: synchronize: ... aclrtSynchronizeEvent(event_), error code is 507011
```

Ascend 侧核心报错如下：

```text
Op execute failed. origin_op_name [RotaryEmbeddingRemovePaddingInterleaved_4]
The DDR address of the MTE instruction is out of range.
fault kernel_name=te_rotaryembeddingremovepaddinginterleaved_...
```

含义：

- `curr_event.synchronize()` 不是根因，只是异步 NPU kernel 错误在同步点暴露。
- 真正失败的 kernel 是 `RotaryEmbeddingRemovePaddingInterleaved_4`。
- `The DDR address of the MTE instruction is out of range` 表示 NPU kernel 搬运 GM/DDR 数据时访问了越界地址。
- 该类问题通常由 kernel 输入元数据不一致导致，例如 `kv_len`、`cumsum_q_len`、`position_ids` 超过 `sin/cos` RoPE 表长度。

## 2. 崩溃现场配置

崩溃分析阶段使用的启动脚本配置为：

```bash
export XPERF_MAX_SEQUENCE_LENGTH=8000
export XLLM_MODEL_MAX_CONTEXT_LENGTH=8000
export XLLM_MAX_NEW_TOKENS=1024

export XLLM_PROCESSOR_api_converter_chain__active_converters="CustomModeReasoningParser,Glm47MoeArkToolParser"
export XLLM_PROCESSOR_api_converter_configs__CustomModeReasoningParser__disable_thinking_prefix="</think>"
export XLLM_PROCESSOR_api_converter_configs__CustomModeReasoningParser__enable_thinking_prefix="<think>"

export XLLM_INFERENCER_SPEC_DECODING_METHOD=mtp
export XLLM_INFERENCER_SPEC_DECODING_STEPS=3
export XLLM_INFERENCER_FORCE_ALIGN_SPEC_Q_LEN=1
```

注意：当前 `/data01/ascend2/mtp_base_add_fix.sh` 中窗口配置已经被改成更大的值，例如：

```bash
export XPERF_MAX_SEQUENCE_LENGTH=202752
export XLLM_MODEL_MAX_CONTEXT_LENGTH=202752
export XLLM_MAX_NEW_TOKENS=131072
```

因此复盘崩溃时，需要以当时日志和 dump 里的实际 tensor shape 为准，而不是只看当前脚本内容。

## 3. Dump 证据

报错 dump 文件：

```text
/data01/ascend2/extra-info/data-dump/14/te_rotaryembeddingremovepaddinginterleaved_48102aaea447c6adb7b99f0f72a600f9064e1309dc31c06be639f28c9221f9bd__kernel0_host.o
```

结合 dump 解析，`RotaryEmbeddingRemovePaddingInterleaved` 的关键输入可映射为：

```text
index 0: q_pe input
index 1: sin
index 2: cos
index 3: kv_len
index 4: cumsum_q_len
index 5: interleaved_offset
index 6: mul_aux
index 7: output
```

dump 中解析到：

```text
kv_len = [13653, 5030, 0, 0, 0]
cumsum_q_len = [0, 4, 8, 8, 8, 8]
```

同时 `sin/cos` 的 dump 大小为：

```text
2048000 bytes
```

按 `bf16` 和 hidden rotary 维度 `128` 计算：

```text
2048000 / 2 / 128 = 8000
```

即 `sin/cos` RoPE 表长度只有 `8000`，最大合法索引是 `7999`。

而 kernel 收到的第一条请求：

```text
kv_len[0] = 13653
```

因此 `RotaryEmbeddingRemovePaddingInterleaved` 在根据 `kv_len` 访问 `sin/cos` 时会越界，最终触发：

```text
The DDR address of the MTE instruction is out of range
```

## 4. 代码链路

### 4.1 NPU 错误暴露点

Python 栈中报错位置在：

```text
/data01/ichaoren/xpu_gpt/xpu_gpt/adapter/npu_xllm/spec_decoding.py
```

核心行为是：

```python
curr_event.synchronize()
```

这里等待 NPU event，同步时暴露之前异步提交的 kernel 错误。

### 4.2 Rotary kernel 调用点

相关代码在：

```text
/data01/ichaoren/xpu_gpt/xpu_gpt/backend/npu/operators/position_emb/rotary_mla.py
```

调用形式为：

```python
q_pe = self.rope_op(
    q_pe,
    fwd_params.sin,
    fwd_params.cos,
    fwd_params.kv_len,
    fwd_params.cumsum_q_len,
    self.interleaved_offset_npu,
    self.mul_aux_npu,
    self.q_input_head_num,
    0,
)
```

也就是说该 kernel 不只依赖 `position_ids`，还直接依赖 `kv_len` 和 `cumsum_q_len`。

### 4.3 gen_ctx 构造中的漏保护点

相关代码在：

```text
/data01/ichaoren/xpu_gpt/xpu_gpt/adapter/npu_xllm/npu_generate_context_builder.py
```

decode 阶段当前逻辑类似：

```python
cur_kv_len = query.sequence.prompt_len + query.sequence.output_len - 1

if self.gen_ctx.enable_spec_decoding:
    cur_pos_ids = [
        min(cur_kv_len + i, self.gen_ctx.max_position_embeddings - self.gen_ctx.spec_decoding_steps - 1)
        for i in range(self.gen_ctx.spec_decoding_steps + 1)
    ]
else:
    cur_pos_ids = [cur_kv_len]

self.all_kv_len[idx] = cur_kv_len
```

这里只 clamp 了 `position_ids`，没有 clamp 或提前拦截 `cur_kv_len/all_kv_len`。

因此即使 `position_ids` 被限制在 `8000` 以内，`kv_len=13653` 仍会进入 `RotaryEmbeddingRemovePaddingInterleaved`，导致 kernel 用非法 `kv_len` 访问 `sin/cos`。

## 5. 为什么 XLLM_MAX_NEW_TOKENS=1024 没有挡住

结论：

```text
XLLM_MAX_NEW_TOKENS=1024 是生效的，但在当前 reasoning/thinking 模式下，它限制的是非 thinking 输出，不限制总生成 token。
```

请求参数构造位置：

```text
/data01/ascend/xLLM/xllm/service/rpc/ark/utils/generate_params_builder.py
```

关键代码：

```python
max_new_tokens = self._parameter_value("max_new_tokens", req, -1)
max_new_tokens_from_env = int(env_config.get("xllm_max_new_tokens", 8192))
max_new_tokens = min(max_new_tokens, max_new_tokens_from_env) if max_new_tokens > 0 else max_new_tokens_from_env

thinking_budget = self._parameter_value("thinking_budget", req, 32 * 1024)
```

说明：

- 如果请求没有传 `max_new_tokens`，会使用环境变量 `XLLM_MAX_NEW_TOKENS`。
- 如果请求传了 `max_new_tokens`，会取请求值和环境变量的较小值。
- 如果请求没有传 `thinking_budget`，默认值是 `32 * 1024 = 32768`。

输出停止判断位置：

```text
/data01/ascend/xLLM/xllm/backend/processor/output_processor.py
```

关键代码：

```python
if (
    query.generate_params.max_completion_tokens is None
    and 0 < query.generate_params.max_new_tokens <= query.sequence.output_len - query.sequence.thinking_len
):
    finished = True
    finished_reason = FinishedReason.LENGTH
```

这里比较的是：

```text
output_len - thinking_len
```

而不是：

```text
output_len
```

因此在 thinking 阶段，如果模型一直没有输出 `</think>`，可能出现：

```text
output_len = 12591
thinking_len = 12591
output_len - thinking_len = 0
max_new_tokens = 1024
```

此时停止条件：

```text
1024 <= 0
```

不成立，请求会继续生成。

但对 NPU RoPE kernel 来说，thinking token 同样占用真实序列位置。硬件看到的是实际 `kv_len`，不会因为 token 是 thinking 就不占位置。

因此在 `thinking_budget=32768`、RoPE 表长度只有 `8000` 的情况下，请求可能先到达 `kv_len > 8000`，然后在 `RotaryEmbeddingRemovePaddingInterleaved` 中越界崩溃。

## 6. 根因

根因可以归纳为三层：

1. 启用了 reasoning/thinking 模式，`XLLM_MAX_NEW_TOKENS` 只限制非 thinking 输出。
2. `thinking_budget` 默认值为 `32768`，大于崩溃现场的 `XLLM_MODEL_MAX_CONTEXT_LENGTH=8000`。
3. MTP async decode 构造下一轮 NPU 输入前，缺少硬性上下文保护，导致 `kv_len=13653` 进入 `RotaryEmbeddingRemovePaddingInterleaved`。

最终表现为：

```text
kv_len = 13653
sin/cos length = 8000
```

kernel 访问 RoPE 表越界，触发 NPU AICore DDR out-of-range。

## 7. 解决方案

### 7.1 推荐方案：增加 forward 前硬保护

在构造 NPU gen_ctx 前增加不区分 thinking/non-thinking 的硬上下文保护：

```text
cur_kv_len + spec_decoding_steps < max_position_embeddings
```

对崩溃现场配置，即：

```text
cur_kv_len + 3 < 8000
```

建议位置：

```text
/data01/ichaoren/xpu_gpt/xpu_gpt/adapter/npu_xllm/npu_generate_context_builder.py
```

在 decode 阶段计算出 `cur_kv_len` 后，进入 `self.all_kv_len[idx] = cur_kv_len` 和后续 tensor 构造前进行检查。

短期防御可以直接抛清晰错误：

```python
max_pos = self.gen_ctx.max_position_embeddings
spec_steps = self.gen_ctx.spec_decoding_steps if self.gen_ctx.enable_spec_decoding else 0
if cur_kv_len + spec_steps >= max_pos:
    raise ValueError(
        f"kv_len exceeds max_position_embeddings before rotary: "
        f"cur_kv_len={cur_kv_len}, spec_steps={spec_steps}, max_position_embeddings={max_pos}"
    )
```

更完整的生产修复应当在上层调度或 output processor 中将该请求标记为 `FinishedReason.LENGTH`，避免 worker 内抛异常导致整个服务退出。

### 7.2 同步约束 thinking_budget

如果当前模型/RoPE 表只支持 `8000`，应避免默认 `thinking_budget=32768` 带来的超上下文风险。

建议策略：

```text
effective_thinking_budget <= max_position_embeddings - prompt_len - spec_decoding_steps - safety_margin
```

或者在请求入口侧将未显式传入的 `thinking_budget` 限制到模型上下文窗口内。

### 7.3 配置一致性检查

如果启动脚本将：

```bash
export XLLM_MODEL_MAX_CONTEXT_LENGTH=202752
```

则需要确认以下配置和模型产物同步支持该长度：

- RoPE `sin/cos` 表是否实际生成到 `202752`。
- `MAX_POSITION` 是否与模型和 rope cache 一致。
- `XLLM_MODEL_NUM_BLOCKS * XLLM_MODEL_SLOT_BLOCK_SIZE` 是否足够覆盖目标上下文长度。

当前脚本中：

```bash
export XLLM_MODEL_NUM_BLOCKS=200
export XLLM_MODEL_SLOT_BLOCK_SIZE=128
```

理论 KV block capacity 只有：

```text
200 * 128 = 25600 tokens
```

如果上下文窗口配置为 `202752`，则还需要重新评估 KV cache block 配置，否则可能从 RoPE 越界转变为 KV cache 容量或 page table 相关问题。

## 8. 不推荐方案

### 8.1 不推荐只 clamp kv_len

不建议在 rotary 前简单将：

```text
kv_len = min(kv_len, max_position_embeddings - 1)
```

原因：

- 可以避免 kernel crash。
- 但会让超长 token 复用错误的 RoPE position。
- 模型语义和精度不可信。
- 问题会被隐藏，后续 attention/KV 仍可能不一致。

### 8.2 不推荐仅依赖 XLLM_MAX_NEW_TOKENS

`XLLM_MAX_NEW_TOKENS` 在 thinking 模式下限制的是非 thinking token，不是总 token。

因此它不能作为 NPU kernel 安全边界。

硬件安全边界必须基于真实序列长度：

```text
prompt_len + output_len
```

而不是：

```text
output_len - thinking_len
```

## 9. 结果验证

### 9.1 已完成的验证

通过 dump 和日志完成了静态证据闭环：

- 报错 kernel 是 `RotaryEmbeddingRemovePaddingInterleaved_4`。
- dump 中 `sin/cos` 表长度为 `8000`。
- dump 中 `kv_len[0] = 13653`。
- 日志中对应请求生成到了 `computed_length≈13652`。
- `kv_len` 大于 `sin/cos` 表长度，能够解释 DDR out-of-range。

### 9.2 修复后建议验证项

修复后建议进行以下验证：

1. 复跑原始 workload，确认不再出现：

```text
RotaryEmbeddingRemovePaddingInterleaved_4
The DDR address of the MTE instruction is out of range
aclrtSynchronizeEvent(event_), error code is 507011
```

2. 打开轻量日志，记录接近边界的请求：

```text
request_id
prompt_len
output_len
thinking_len
cur_kv_len
spec_decoding_steps
max_position_embeddings
```

3. 验证请求在进入 NPU forward 前被正确截断或完成：

```text
cur_kv_len + spec_decoding_steps < max_position_embeddings
```

4. 验证超长 thinking 请求不会导致服务崩溃，而是以 `LENGTH` 或明确错误返回。

5. 如果调整了 `XLLM_MODEL_MAX_CONTEXT_LENGTH`，需要确认 RoPE 表 dump 或运行日志中的 `sin/cos` 长度与配置一致。

## 10. 一句话总结

这次崩溃不是随机 NPU 故障，而是输入越界：reasoning/thinking 模式下 `XLLM_MAX_NEW_TOKENS=1024` 没有限制总生成长度，默认 `thinking_budget=32768` 允许 thinking token 继续增长；MTP async 路径在 NPU forward 前缺少真实 `kv_len` 硬保护，最终把 `kv_len=13653` 传给了只有 `8000` 行 `sin/cos` 的 RoPE kernel，导致 `RotaryEmbeddingRemovePaddingInterleaved` 访问 DDR 越界并使服务崩溃。
