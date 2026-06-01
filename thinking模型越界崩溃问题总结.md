# RotaryEmbeddingRemovePaddingInterleaved 越界崩溃问题总结

## 1. 问题现象

服务运行过程中在 NPU 上发生 AICore 异常，主进程随后触发 shutdown。应用侧看到的直接报错是：

```text
RuntimeError: synchronize: ... aclrtSynchronizeEvent(event_), error code is 507011
```

Ascend 侧日志显示真正失败的 NPU kernel 是 `RotaryEmbeddingRemovePaddingInterleaved_4`：

```text
Op execute failed. origin_op_name [RotaryEmbeddingRemovePaddingInterleaved_4]
The DDR address of the MTE instruction is out of range.
fault kernel_name=te_rotaryembeddingremovepaddinginterleaved_...
```

这里的 `curr_event.synchronize()` 只是异步错误暴露点，不是根因。真正含义是：`RotaryEmbeddingRemovePaddingInterleaved_4` 在搬运 GM/DDR 数据时访问了非法地址，导致设备侧报 `The DDR address of the MTE instruction is out of range`。

## 2. 问题分析

出问题时，启动脚本中的窗口和输出长度配置为：

```bash
export XPERF_MAX_SEQUENCE_LENGTH=8000
export XLLM_MODEL_MAX_CONTEXT_LENGTH=8000
export XLLM_MAX_NEW_TOKENS=1024
```

同时服务启用了 reasoning/thinking parser 和 MTP：

```bash
export XLLM_PROCESSOR_api_converter_chain__active_converters="CustomModeReasoningParser,Glm47MoeArkToolParser"
export XLLM_PROCESSOR_api_converter_configs__CustomModeReasoningParser__disable_thinking_prefix="</think>"
export XLLM_PROCESSOR_api_converter_configs__CustomModeReasoningParser__enable_thinking_prefix="<think>"

export XLLM_INFERENCER_SPEC_DECODING_METHOD=mtp
export XLLM_INFERENCER_SPEC_DECODING_STEPS=3
export XLLM_INFERENCER_FORCE_ALIGN_SPEC_Q_LEN=1
```

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

dump 中关键数据如下：

```text
kv_len = [13653, 5030, 0, 0, 0]
cumsum_q_len = [0, 4, 8, 8, 8, 8]
sin/cos dump size = 2048000 bytes
```

按 `bf16` 和 rotary hidden 维度 `128` 计算，`sin/cos` RoPE 表长度为：

```text
2048000 / 2 / 128 = 8000
```

即事故现场的 `sin/cos` 最大合法位置是 `7999`，但 kernel 收到的第一条有效请求已经是：

```text
kv_len[0] = 13653
```

因此，设备侧访问 `sin/cos[13653]` 附近时必然越界，这是本次 AICore DDR out-of-range 的直接原因。

从代码链路看，Python 栈中的报错位置在：

```text
/data01/ichaoren/xpu_gpt/xpu_gpt/adapter/npu_xllm/spec_decoding.py
```

核心行为是：

```python
curr_event.synchronize()
```

这里等待 NPU event，同步时暴露之前异步提交的 kernel 错误。实际出错的 rotary 调用在：

```text
/data01/ichaoren/xpu_gpt/xpu_gpt/backend/npu/operators/position_emb/rotary_mla.py
```

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

这个 kernel 不只依赖 `position_ids`，也直接依赖 `kv_len` 和 `cumsum_q_len`。而 `kv_len` 在 gen_ctx 构造阶段来自：

```text
/data01/ichaoren/xpu_gpt/xpu_gpt/adapter/npu_xllm/npu_generate_context_builder.py
```

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

这里的问题是：代码只对 `position_ids` 做了 clamp，没有在 NPU forward 前阻止超长 `cur_kv_len/all_kv_len` 继续进入 rotary kernel。所以即使 `position_ids` 被限制在 `8000` 附近，`kv_len=13653` 仍会传给 `RotaryEmbeddingRemovePaddingInterleaved`。

进一步看为什么 `XLLM_MAX_NEW_TOKENS=1024` 没有挡住。请求参数构造位置在：

```text
/data01/ascend/xLLM/xllm/service/rpc/ark/utils/generate_params_builder.py
```

```python
max_new_tokens = self._parameter_value("max_new_tokens", req, -1)
max_new_tokens_from_env = int(env_config.get("xllm_max_new_tokens", 8192))
max_new_tokens = min(max_new_tokens, max_new_tokens_from_env) if max_new_tokens > 0 else max_new_tokens_from_env

thinking_budget = self._parameter_value("thinking_budget", req, 32 * 1024)
```

说明 `XLLM_MAX_NEW_TOKENS=1024` 是生效的，但 `thinking_budget` 如果请求未传，会使用默认值 `32768`。输出停止判断位置在：

```text
/data01/ascend/xLLM/xllm/backend/processor/output_processor.py
```

```python
if (
    query.generate_params.max_completion_tokens is None
    and 0 < query.generate_params.max_new_tokens <= query.sequence.output_len - query.sequence.thinking_len
):
    finished = True
    finished_reason = FinishedReason.LENGTH
```

关键点是这里比较的是：

```text
output_len - thinking_len
```

不是总 `output_len`。因此在 thinking 阶段，如果模型一直没有输出 `</think>`，可能出现：

```text
output_len = 12591
thinking_len = 12591
output_len - thinking_len = 0
max_new_tokens = 1024
```

此时 `1024 <= 0` 不成立，请求不会因为 `XLLM_MAX_NEW_TOKENS=1024` 停止。对 NPU RoPE kernel 来说，thinking token 仍然占用真实序列位置，因此真实 `kv_len` 可以继续增长并超过 RoPE 表长度。

最终根因可以归纳为：

1. 启用了 reasoning/thinking 模式，`XLLM_MAX_NEW_TOKENS` 只限制非 thinking 输出。
2. `thinking_budget` 默认值为 `32768`，大于崩溃现场的 `XLLM_MODEL_MAX_CONTEXT_LENGTH=8000`。
3. MTP async decode 构造下一轮 NPU 输入前，缺少硬性上下文保护，导致 `kv_len=13653` 进入 `RotaryEmbeddingRemovePaddingInterleaved`。

最终表现为：

```text
kv_len = 13653
sin/cos length = 8000
```

kernel 访问 RoPE 表越界，触发 NPU AICore DDR out-of-range。

## 3. 解决方案

根据上面的分析，问题不是 `XLLM_MAX_NEW_TOKENS=1024` 没有读取，而是 `8000` 的 RoPE 上下文窗口无法覆盖实际 reasoning/thinking 路径产生的真实序列长度。因此本次实际采用的解决方案是同步扩大窗口和输出长度配置：

```bash
export XPERF_MAX_SEQUENCE_LENGTH=202752
export XLLM_MODEL_MAX_CONTEXT_LENGTH=202752
export XLLM_MAX_NEW_TOKENS=131072
```

这组配置从三个层面解决问题：

- `XLLM_MODEL_MAX_CONTEXT_LENGTH=202752`：使模型上下文窗口和 RoPE `sin/cos` cache 理论上不再停留在 `8000`，从而让 `kv_len≈13653` 进入合法范围。
- `XPERF_MAX_SEQUENCE_LENGTH=202752`：使性能/调度侧看到的最大序列长度与模型窗口保持一致，避免仍按旧窗口估算。
- `XLLM_MAX_NEW_TOKENS=131072`：使默认输出长度配置与长上下文 reasoning 场景一致，不再用 `1024` 描述当前运行目标。

该方案本质是“扩大合法运行区间”：原来 `kv_len=13653` 已超过 `8000` 的 RoPE 表，现在通过扩展窗口让这类长度落在模型可支持范围内。

## 4. 总结

这次崩溃不是随机 NPU 故障，而是输入越界。事故发生时 `XLLM_MODEL_MAX_CONTEXT_LENGTH=8000`，RoPE `sin/cos` 表也只有 `8000` 行；reasoning/thinking 模式下 `XLLM_MAX_NEW_TOKENS=1024` 只限制非 thinking 输出，真实序列长度仍继续增长，最终 `kv_len=13653` 被传入 `RotaryEmbeddingRemovePaddingInterleaved`，触发 DDR 越界。根据这一分析，本次处置将窗口和输出长度配置同步扩大到 `202752/202752/131072`，使事故请求长度进入合法区间；后续需要验证 RoPE、KV cache 和调度配置是否全链路一致，并建议增加 forward 前真实 `kv_len` 硬保护作为兜底。
