# PD 分离（Disaggregated Prefill）核心处理流程分析

> 硬件平台：Ascend NPU | KV 传输：Mooncake TransferEngine（RDMA）  
> 涉及文件：  
> - `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`  
> - `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`  
> - `vllm/v1/core/sched/scheduler.py`（上游调度器）  
> - `vllm/v1/worker/gpu/kv_connector.py`（上游 Worker 侧 KV Connector 封装）  
> - `vllm_ascend/worker/model_runner_v1.py`（NPU ModelRunner）

---

## 1. 系统架构总览

PD 分离系统由三类角色协作完成：

| 角色 | 职责 | 对应组件 |
|------|------|---------|
| **Proxy** | 请求路由、负载均衡、KV 元数据中转 | `load_balance_proxy_server_example.py` |
| **P-Node** | Prefill 计算，生成 KV Cache，等待 D-Node 拉取 | `MooncakeConnectorScheduler` + `KVCacheSendingThread` |
| **D-Node** | 远程拉取 KV Cache，Decode 生成 | `MooncakeConnectorWorker` + `KVCacheRecvingThread` |

核心数据流：

```
Client → Proxy → P-Node (Prefill, max_tokens=1) → Mooncake RDMA → D-Node (Decode) → Client
                  │                                       ↑
                  └── kv_transfer_params ── Proxy ────────┘
```

---

## 2. MooncakeConnector 公开函数清单

`MooncakeConnector` 根据角色（`KVConnectorRole`）在 Scheduler 进程和 Worker 进程中分别实例化，公开函数分属两侧：

### Scheduler 侧（`MooncakeConnectorScheduler` 代理执行）

| 函数 | 作用 | 调用时机 |
|------|------|---------|
| `get_num_new_matched_tokens` | 判断请求需从远程拉取多少 token 的 KV | 请求首次调度时 |
| `update_state_after_alloc` | Block 分配后记录需要 recv 的请求 | Block 分配完成后 |
| `build_connector_meta` | 打包 KV Connector 元数据到 SchedulerOutput | 调度步骤结束时 |
| `request_finished` | 请求完成时打包 KV 传输参数，延迟释放 block | 请求生成完成时 |
| `set_xfer_handshake_metadata` | 保存跨节点 Worker 元数据映射 | 引擎初始化时 |

### Worker 侧（`MooncakeConnectorWorker` 代理执行）

| 函数 | 作用 | 调用时机 |
|------|------|---------|
| `register_kv_caches` | 注册 KV Cache 到 TransferEngine，启动收发线程 | KV Cache 初始化时 |
| `start_load_kv` | 提交 KV 拉取任务到后台线程 | 模型 forward 前 |
| `wait_for_layer_load` | 等待某层 KV 加载完成（**空实现**） | 每层 attention |
| `save_kv_layer` | 保存某层 KV（**空实现**） | 每层 attention |
| `wait_for_save` | 等待所有 save 完成（**空实现**） | 模型 forward 后 |
| `get_finished` | 获取已完成发送/接收的请求 ID | 模型 forward 后 |
| `get_handshake_metadata` | 返回 Worker 握手元数据 | 引擎初始化时 |

---

## 3. 按请求生命周期的完整调用时序

### 阶段 0：引擎初始化（一次性）

#### 0.1 Worker 侧：`register_kv_caches(kv_caches)`

**调用路径**：

```
NPUModelRunner._setup_kv_cache()                    [model_runner_v1.py:L2788-2789]
  → get_kv_transfer_group().register_kv_caches(kv_caches)
    → MooncakeConnector.register_kv_caches()
      → MooncakeConnectorWorker.register_kv_caches()  [mooncake_connector.py:L1184-1286]
```

**内部逻辑**：

1. 检测 KV Cache 形状，判断 MLA / Sparse / MHA 模式
2. 计算 `block_len`（每个 block 的字节数）
3. 将 KV Cache 基地址注册到 Mooncake `TransferEngine`（`global_te.register_buffer`）
4. 构建 `MooncakeAgentMetadata`（包含 engine_id、te_rpc_port、kv_caches_base_addr）
5. **P-Node**（`kv_role == "kv_producer"`）：启动 `KVCacheSendingThread`
   - ZMQ ROUTER socket 监听，端口 = `side_channel_port + device_index`
   - 处理 `GET_META_MSG`（返回元数据）和 `DONE_RECVING_MSG`（释放 block）
6. **D-Node**（`kv_role == "kv_consumer"`）：启动 `KVCacheRecvingThread`
   - 从 `request_queue` 消费拉取任务
   - 通过 ZMQ REQ socket 与 P-Node 通信
   - 执行 Mooncake RDMA 传输

**NPU 特有**：`model_runner_v1.py` 直接调用 `get_kv_transfer_group().register_kv_caches()`，而非像 GPU 路径那样通过 `ActiveKVConnector` 封装。

#### 0.2 Worker 侧：`get_handshake_metadata()`

**调用路径**：

```
EngineCore.__init__()                                [core.py:L161-177]
  → model_executor.get_kv_connector_handshake_metadata()
    → Executor.collective_rpc("get_kv_connector_handshake_metadata")
      → Worker.get_kv_connector_handshake_metadata()
        → MooncakeConnector.get_handshake_metadata()  [mooncake_connector.py:L884-895]
          → MooncakeConnectorWorker.xfer_handshake_metadata
```

**作用**：从所有 Worker 收集 `MooncakeAgentMetadata`，返回给 Scheduler 侧。

#### 0.3 Scheduler 侧：`set_xfer_handshake_metadata(metadata)`

**调用路径**：

```
EngineCore.__init__()                                [core.py:L161-177]
  → kv_connector.set_xfer_handshake_metadata(content)
    → MooncakeConnector.set_xfer_handshake_metadata()  [mooncake_connector.py:L897-905]
      → MooncakeConnectorScheduler.set_xfer_handshake_metadata()  [mooncake_connector.py:L1079-1090]
```

**内部逻辑**：

```python
for local_rank, rank_metadata in metadata.items():
    self.multi_nodes_meta_mapping[str(local_rank)] = {
        "host": rank_metadata.local_ip,
        "engine_id": rank_metadata.engine_id,
    }
```

保存跨节点 Worker 的 host → engine_id 映射，供后续 `request_finished` 打包 `remote_multi_nodes_meta_mapping` 使用。

---

### 阶段 1：Proxy 接收请求并路由

#### 1.1 请求到达 Proxy

```
Client → POST /v1/completions 或 /v1/chat/completions
```

`_handle_completions`（[proxy:L710](file:///data01/ichaoren/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L710)）处理请求。

#### 1.2 两阶段调度：Prefill → Decode

`_handle_select_instance`（[proxy:L659](file:///data01/ichaoren/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L659)）：

**阶段一：选择 P-Node 并执行 Prefill**

```python
# 1. 计算负载分数，选择最空闲的 prefiller
prefiller_score = proxy_state.calculate_prefill_scores(request_length)
prefiller_idx = proxy_state.select_prefiller(prefiller_score)

# 2. 向 P-Node 发送请求，max_tokens=1
response = await send_request_to_service(prefiller.client, prefiller_idx, api, req_data, request_id)
```

`send_request_to_service`（[proxy:L576](file:///data01/ichaoren/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L576)）关键操作：

- **强制 `max_tokens=1`**：P-Node 只做 prefill 计算，不生成完整回复
- **注入 `kv_transfer_params`**：
  ```python
  req_data["kv_transfer_params"] = {
      "do_remote_decode": True,
      "do_remote_prefill": False,
      "remote_engine_id": None,
      "remote_block_ids": None,
      "remote_host": None,
      "remote_port": None,
      "aborted_request": list(aborted_requests),
  }
  ```
- **附带 aborted_request**：采用"懒释放"机制，将之前中断的请求 ID 捎带给同一 prefiller

**阶段二：提取 kv_transfer_params 并选择 D-Node**

```python
# 3. 从 P-Node 响应中提取 kv_transfer_params
kv_transfer_params = response.json().get("kv_transfer_params", {})
req_data["kv_transfer_params"] = kv_transfer_params

# 4. 选择 D-Node
decoder_idx = proxy_state.select_decoder(decoder_score)
```

P-Node 在 `request_finished` 时打包的 `kv_transfer_params` 包含：
- `do_remote_prefill: True` — 告知 D-Node 需要远程拉取
- `remote_block_ids` — P-Node 上的 KV block 编号
- `remote_engine_id`、`remote_host`、`remote_port` — P-Node 标识和地址
- `remote_pcp_size`、`remote_dcp_size`、`remote_ptp_size` — P-Node 并行配置
- `num_prompt_blocks` — prompt 占用的 block 数量
- `remote_multi_nodes_meta_mapping` — 跨节点元数据

---

### 阶段 2：P-Node 处理 Prefill 请求

#### 2.1 请求进入 Scheduler

`EngineCore.add_request()` → `Scheduler.add_request()`，请求携带 `kv_transfer_params`（`do_remote_decode=True`）。

#### 2.2 `get_num_new_matched_tokens(request, num_computed_tokens)`

**调用路径**：

```
Scheduler._schedule()                                [scheduler.py:L607-612]
  → self.connector.get_num_new_matched_tokens(request, num_new_local_computed_tokens)
    → MooncakeConnectorScheduler.get_num_new_matched_tokens()  [mooncake_connector.py:L949-980]
```

**P-Node 逻辑**：

```python
if params.get("do_remote_prefill"):
    # P-Node 本地计算，不需要远程拉取
    count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
    return count, count > 0
return 0, False
```

P-Node 本地执行 prefill，返回需要计算的 token 数量。

#### 2.3 `update_state_after_alloc(request, blocks, num_external_tokens)`

**调用路径**：

```
Scheduler._schedule()                                [scheduler.py:L746-751]
  → self.connector.update_state_after_alloc(request, blocks, num_external_tokens)
    → MooncakeConnectorScheduler.update_state_after_alloc()  [mooncake_connector.py:L982-1003]
```

**P-Node 逻辑**：

- 将请求加入 `_reqs_in_batch` 集合
- P-Node 的 `do_remote_prefill` 为 False（由 Proxy 设置），所以不会加入 `_reqs_need_recv`

#### 2.4 `build_connector_meta(scheduler_output)`

**调用路径**：

```
Scheduler.schedule()                                 [scheduler.py:L912-916]
  → self.connector.build_connector_meta(scheduler_output)
    → MooncakeConnectorScheduler.build_connector_meta()  [mooncake_connector.py:L1005-1031]
```

**P-Node 逻辑**：

- 将 `_reqs_need_recv`（P-Node 通常为空）和 `_reqs_need_send`（延迟释放的请求）打包为 `MooncakeConnectorMetadata`
- 附加到 `scheduler_output.kv_connector_metadata`
- 清空内部状态

#### 2.5 P-Node Worker 侧执行 forward

**NPU 调用路径**：

```
NPUModelRunner.execute_model()                       [model_runner_v1.py:L1496-1501]
  → self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output
    → KVConnectorModelRunnerMixin._get_kv_connector_output()  [kv_connector_model_runner_mixin.py:L95-130]
      → kv_connector.bind_connector_metadata(metadata)     # 绑定 metadata
      → kv_connector.start_load_kv(forward_context)        # P-Node: 标记 delayed_request
      → [模型 forward 执行]
      → kv_connector.wait_for_save()                       # 空实现
      → kv_connector.get_finished()                        # 获取已完成发送的请求
      → kv_connector.clear_connector_metadata()            # 清除 metadata
```

P-Node 的 `start_load_kv` 主要做两件事：
1. 对 `metadata.reqs_in_batch` 中的请求，在 `KVCacheSendingThread` 标记 `add_req_to_process`
2. 对 `metadata.requests_to_send` 中的请求，标记 `add_delayed_request`（延迟释放 block）

#### 2.6 `request_finished(request, block_ids)`

**调用路径**：

```
Scheduler._connector_finished()                      [scheduler.py:L1981-2010]
  → self.connector.request_finished(request, block_ids[0])
    → MooncakeConnectorScheduler.request_finished()  [mooncake_connector.py:L1033-1077]
```

**触发条件**：仅当 `do_remote_decode=True` 且 `request.status == FINISHED_LENGTH_CAPPED`。

**P-Node 核心出口逻辑**：

```python
computed_block_ids = block_ids
delay_free_blocks = len(computed_block_ids) > 0
if delay_free_blocks:
    self._reqs_need_send[request.request_id] = time.time()

num_prompt_blocks = math.ceil(len(request.prompt_token_ids) / self.block_size)

return delay_free_blocks, dict(
    do_remote_prefill=True,
    do_remote_decode=False,
    remote_block_ids=computed_block_ids,
    remote_engine_id=self.engine_id,
    remote_request_id=request.request_id,
    remote_host=self.side_channel_host,
    remote_port=self.side_channel_port,
    remote_pcp_size=self.pcp_size,
    remote_dcp_size=self.dcp_size,
    remote_ptp_size=self.tp_size,
    last_token_id=request.output_token_ids[-1],
    remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
    num_prompt_blocks=num_prompt_blocks,
)
```

**关键设计**：
- `delay_free_blocks=True`：P-Node 的 KV block 不立即释放，等 D-Node 拉取完再释放
- `kv_transfer_params` 通过 vLLM Engine 的输出返回给 Proxy，Proxy 再转发给 D-Node

---

### 阶段 3：D-Node 处理 Decode 请求

#### 3.1 请求进入 D-Node Scheduler

Proxy 将 P-Node 返回的 `kv_transfer_params` 注入请求，发送给 D-Node。此时 `do_remote_prefill=True`。

#### 3.2 `get_num_new_matched_tokens(request, num_computed_tokens)`

**调用路径**：同 2.2。

**D-Node 逻辑**：

```python
if params.get("do_remote_prefill"):
    count = max(len(request.prompt_token_ids) - num_computed_tokens, 0)
    return count, count > 0  # load_kv_async=True
```

D-Node 返回需要远程拉取的 token 数量，`load_kv_async=True` 表示异步加载。

**影响**：请求进入 `WAITING_FOR_REMOTE_KVS` 状态，等待 KV 拉取完成后才能调度执行。

#### 3.3 `update_state_after_alloc(request, blocks, num_external_tokens)`

**调用路径**：同 2.3。

**D-Node 逻辑**：

```python
if params.get("do_remote_prefill"):
    if params.get("remote_block_ids"):
        local_block_ids = blocks.get_unhashed_block_ids()
        self._reqs_need_recv[request.request_id] = (request, local_block_ids, num_external_tokens)
    params["do_remote_prefill"] = False  # 只触发一次 KV 传输
```

将请求加入 `_reqs_need_recv`，等待 `build_connector_meta` 打包给 Worker。

#### 3.4 `build_connector_meta(scheduler_output)`

**调用路径**：同 2.4。

**D-Node 逻辑**：

将 `_reqs_need_recv` 中的请求转换为 `MooncakeConnectorMetadata`，每个请求生成一个 `ReqMeta`：

```python
ReqMeta(
    local_block_ids=block_ids,
    num_external_tokens=num_external_tokens,
    remote_block_ids=kv_transfer_params["remote_block_ids"],
    remote_engine_id=kv_transfer_params["remote_engine_id"],
    remote_request_id=kv_transfer_params["remote_request_id"],
    remote_host=kv_transfer_params["remote_host"],
    remote_port=kv_transfer_params["remote_port"],
    remote_pcp_size=kv_transfer_params.get("remote_pcp_size", 1),
    remote_dcp_size=kv_transfer_params.get("remote_dcp_size", 1),
    remote_ptp_size=kv_transfer_params.get("remote_ptp_size"),
    remote_multi_nodes_meta_mapping=kv_transfer_params.get("remote_multi_nodes_meta_mapping", {}),
    num_prompt_blocks=kv_transfer_params.get("num_prompt_blocks", 0),
)
```

#### 3.5 D-Node Worker 侧执行 KV 拉取

**NPU 调用路径**：

```
NPUModelRunner.execute_model()                       [model_runner_v1.py:L1496-1501]
  → self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output
    → KVConnectorModelRunnerMixin._get_kv_connector_output()
      → kv_connector.bind_connector_metadata(metadata)
      → kv_connector.start_load_kv(forward_context)    # D-Node 核心：提交拉取任务
```

#### 3.6 `start_load_kv(metadata)` — D-Node 核心入口

**调用路径**：

```
ActiveKVConnector.pre_forward()                      [kv_connector.py:L74]
  → self.kv_connector.start_load_kv(get_forward_context())
    → MooncakeConnectorWorker.start_load_kv()         [mooncake_connector.py:L1520-1606]
```

**内部逻辑**：

**无 CP 场景**（`pcp_size * dcp_size == 1`）：

```python
# 1. 基于 req_id 哈希选择 P-Node 的 TP rank
chosen_rank_list = self._get_remote_rank(remote_req_id, prefill_tp_size)
remote_handshake_port_list = [[x + meta.remote_port] for x in chosen_rank_list]

# 2. 为每个 (PP stage, TP slice) 提交一个拉取任务
for i in range(tp_num_need_pulls * self._prefill_pp_size):
    self.kv_recv_thread.add_request(
        request_id=req_id,
        offset=i,              # 全局偏移
        tp_num_need_pulls=tp_num_need_pulls,
        all_task_done=(i == tp_num_need_pulls * self._prefill_pp_size - 1),
    )
```

**有 CP 场景**（`pcp_size * dcp_size > 1`）：

调用 `_get_kv_split_metadata` 计算 CP 分组下的 port 映射，然后按 `pcp_dcp_rank × tp_num_need_pulls` 提交拉取任务。

#### 3.7 远程 Rank 选路：`_get_remote_ranks_for_req`

**调用路径**：`start_load_kv` → `_get_remote_rank` → `_get_remote_ranks_for_req`（[mooncake_connector.py:L1668](file:///data01/ichaoren/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L1668)）

**核心算法**：

```python
seed = string_to_int64_hash(req_id)  # SHA-256 → int64
rand = random.Random(seed)
ori_data = np.arange(prefill_tp_size * prefill_pp_size)
ori_data = ori_data.reshape(prefill_pp_size, -1)
num_groups = max(1, len(ori_data[0]) // num_kv_head)
rand_group_index = rand.sample(range(num_groups), max(decode_tp_size // num_kv_head, 1))
```

- **MLA/Sparse**：`tp_num_need_pulls=1`，每个 D-rank 只选 1 个 P-rank
- **MHA/GQA**：`tp_num_need_pulls>1`，每个 D-rank 需选多个 P-rank 的 head 分片
- **稳定随机**：相同 `req_id` 总是选择相同的 P-rank 组合

#### 3.8 KV Cache 拉取：`_transfer_kv_cache`

**调用路径**：`KVCacheRecvingThread._handle_request` → `_transfer_kv_cache`（[mooncake_connector.py:L469](file:///data01/ichaoren/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L469)）

**Step 1：计算偏移量**

```python
global_offset = offset
prefill_pp_rank = offset // tp_num_need_pulls   # 属于哪个 PP stage
inner_offset = offset % tp_num_need_pulls       # 该 PP stage 内的第几个 TP 分片
```

**Step 2：获取远程元数据**（首次连接时通过 ZMQ 获取）

```python
if remote_engine_id not in self.kv_caches_base_addr:
    self._get_remote_metadata(remote_host, remote_handshake_port)
    # ZMQ REQ → P-Node ROUTER → 返回 MooncakeAgentMetadata
```

**Step 3：构建 RDMA 传输列表**

```python
for k, (src_layer_base_addr, dst_layer_base_addr) in enumerate(zip(local_addrs, remote_addrs)):
    block_len = self.block_len[k % block_length]
    inner_block_len = block_len // tp_num_need_pulls
    for remote_block_id, local_block_id in zip(grouped_remote_block_ids, grouped_local_block_ids):
        src = src_layer_base_addr + local_block_id[0] * block_len + inner_offset * inner_block_len
        dst = dst_layer_base_addr + remote_block_id[0] * inner_block_len
        length = inner_block_len * len(local_block_id)
```

- `src`：D-Node 本地 KV Cache 的目标地址（含 `inner_offset` 偏移写入）
- `dst`：P-Node 远程 KV Cache 的源地址（每个 P-rank 只占 `inner_block_len` 大小）

**Step 4：Mooncake RDMA 传输**

```python
ret = self.engine.batch_transfer_sync_read(session_id, src_list, dst_list, length_list)
```

**Step 5：reformat_kv_cache（仅 MHA/GQA 且 TP 不对等）**

```python
is_kv_transfer_end = global_offset == tp_num_need_pulls * prefill_pp_size - 1
need_cat_cache = tp_num_need_pulls > 1 and is_kv_transfer_end
need_nz_cache = get_ascend_config().enable_kv_nz and is_kv_transfer_end
```

当所有 TP 分片拉取完成后：

- **`need_cat_cache`**（MHA/GQA TP 不对等）：调用 `_cat_kv_cache`
  1. `npu_paged_cache_load` 将 block 数据读入连续 buffer
  2. `transpose_(1, 2)` 将 `[block, tp_slice, token, head]` 重排为 `[block, token, head, dim]`
  3. `_npu_reshape_and_cache` 写回 paged KV Cache
- **`need_nz_cache`**（NPU NZ 格式）：调用 `_nz_kv_cache` 转换内存布局
- **NPU 融合算子**：若 `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK=1`，使用 `torch.ops._C_ascend.transpose_kv_cache_by_block` 一次性完成

#### 3.9 完成通知与 Block 释放

拉取完成后，D-Node 通过 ZMQ 向 P-Node 发送 `DONE_RECVING_MSG`：

```
KVCacheRecvingThread._handle_request()               [mooncake_connector.py:L443-453]
  → _send_done_signal_to_free_remote_port()          # 释放远程 port 计数
  → _send_done_recv_signal()                          # ZMQ REQ → P-Node ROUTER
```

P-Node 收到后释放对应的 KV block：

```
KVCacheSendingThread.run_busy_loop()                  [mooncake_connector.py:L271-285]
  → DONE_RECVING_MSG:
    → task_tracker.update_done_task_count(request_id)  # 标记请求完成
    → send ACK                                         # 确认收到
```

#### 3.10 `get_finished(finished_req_ids)`

**调用路径**：

```
ActiveKVConnector.post_forward()                     [kv_connector.py:L91-92]
  → self.kv_connector.get_finished(scheduler_output.finished_req_ids)
    → MooncakeConnectorWorker.get_finished()          [mooncake_connector.py:L1288-1307]
```

**NPU 调用路径**：

```
NPUModelRunner.execute_model()                       [model_runner_v1.py:L1496-1501]
  → _get_kv_connector_output() → kv_connector.get_finished()
```

**返回值**：
- P-Node：`(done_sending, set())` — 已完成发送（收到 `DONE_RECVING_MSG`）的请求 ID
- D-Node：`(set(), done_recving)` — 已完成接收的请求 ID

**后续处理**（Scheduler 侧）：

```
Scheduler._update_from_kv_xfer_finished()            [scheduler.py:L2083-2110]
  → finished_recving: 请求从 WAITING_FOR_REMOTE_KVS → WAITING，可被调度执行
  → finished_sending: 释放 P-Node 的 KV block
```

---

### 阶段 4：D-Node 执行 Decode 并流式返回

#### 4.1 D-Node 正常 Decode

KV 拉取完成后，请求回到 `WAITING` 状态，被正常调度执行 Decode。

#### 4.2 Proxy 流式转发

`generate_stream`（[proxy:L730](file:///data01/ichaoren/vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py#L730)）从 D-Node 流式读取解码结果：

- **首次收到 chunk 时释放 P-Node 的 KV Cache 计量**：`release_prefiller_kv`
- **异常时标记 abort**：中断的请求 ID 会在下次调度到同一 prefiller 时捎带通知
- **流结束后释放 D-Node 计量**：`release_decoder`

---

## 4. tp_num_need_pulls 计算机制

```python
# mooncake_connector.py:L1153-1158
if is_deepseek_mla:
    tp_num_need_pulls = 1
else:
    num_d_block_heads = max(1, num_kv_heads // decode_tp_size)
    num_p_block_heads = max(1, num_kv_heads // prefill_tp_size)
    tp_num_need_pulls = num_d_block_heads // num_p_block_heads
```

| 模型类型 | P-TP | D-TP | num_kv_heads | tp_num_need_pulls | reformat |
|---------|------|------|-------------|-------------------|----------|
| MLA | 4 | 2 | 1（逻辑） | 1 | 否 |
| MHA | 4 | 2 | 32 | 2 | 是 |
| MHA | 4 | 4 | 32 | 1 | 否 |
| GQA | 8 | 2 | 16 | 4 | 是 |

---

## 5. offset 与 inner_offset 映射关系

在 `start_load_kv` 中，D-Node 为每个 `(PP stage, TP slice)` 组合提交一个拉取任务，`offset` 是全局递增索引：

```
offset = 0:  pp_rank=0, inner_offset=0  (PP stage 0, TP slice 0)
offset = 1:  pp_rank=0, inner_offset=1  (PP stage 0, TP slice 1)
offset = 2:  pp_rank=1, inner_offset=0  (PP stage 1, TP slice 0)
offset = 3:  pp_rank=1, inner_offset=1  (PP stage 1, TP slice 1)
```

在 `_transfer_kv_cache` 中：

```python
prefill_pp_rank = offset // tp_num_need_pulls
inner_offset = offset % tp_num_need_pulls
```

- `prefill_pp_rank`：确定拉取哪一层的 KV Cache（`first_layer_index, end_layer_index = pp_layer_indices[prefill_pp_rank]`）
- `inner_offset`：确定在 block 内的偏移位置（`src = base + inner_offset * inner_block_len`）

---

## 6. ZMQ Side Channel 通信协议

P-Node 的 `KVCacheSendingThread` 作为 ZMQ ROUTER 监听，D-Node 的 `KVCacheRecvingThread` 通过 ZMQ REQ 连接。

### 消息类型

| 消息 | 方向 | 内容 | 响应 |
|------|------|------|------|
| `GET_META_MSG` | D→P | `(GET_META_MSG, "")` | `MooncakeAgentMetadata`（msgpack 编码） |
| `DONE_RECVING_MSG` | D→P | `(DONE_RECVING_MSG, request_id, remote_port_send_num)` | `ACK` |

### MooncakeAgentMetadata 结构

```python
class MooncakeAgentMetadata(msgspec.Struct):
    engine_id: str
    te_rpc_port: int
    kv_caches_base_addr: list[int]
    num_blocks: int
    local_ip: str
```

---

## 7. Proxy 负载均衡策略

### P-Node 优先级

```python
priority = server.active_tokens + server.active_kv_cache * 0.3
```

KV Cache 占用也纳入考量，避免 P-Node 内存不均。

### D-Node 优先级

```python
priority = server.active_tokens
```

### 懒释放 Abort 机制

```python
# 标记 abort
proxy_state.abort_prefiller_request(prefiller_idx, request_id)

# 捎带到下次请求
aborted_requests = proxy_state.acquire_aborted_prefiller_requests(prefiller_idx)
req_data["kv_transfer_params"]["aborted_request"] = list(aborted_requests)
```

中断的请求 ID 不会立即通知 P-Node，而是在下次调度到同一 prefiller 时捎带在请求中，减少网络开销。

---

## 8. Mooncake 设计特点总结

| 特点 | 说明 |
|------|------|
| **Pull 模式** | D-Node 主动通过 RDMA 从 P-Node 拉取 KV Cache，而非 P-Node 推送 |
| **Step 级异步** | `wait_for_layer_load`/`save_kv_layer`/`wait_for_save` 全部空实现，KV 传输在 step 级别异步执行 |
| **延迟释放** | P-Node 的 block 在 `request_finished` 时标记延迟释放，收到 `DONE_RECVING_MSG` 后才真正释放 |
| **元数据双通道** | 调度信息走 `SchedulerOutput.kv_connector_metadata`（Scheduler→Worker），KV 传输元数据走 ZMQ side channel（Worker↔Worker） |
| **稳定随机选路** | 基于 `req_id` SHA-256 哈希的确定性选路，相同请求总是选择相同的 P-rank 组合 |
| **NPU 融合算子** | `transpose_kv_cache_by_block` 一次性完成 MHA reformat，避免多次 NPU 同步 |
| **NZ 格式支持** | `enable_kv_nz` 时自动转换 KV Cache 为 NPU NZ 内存布局 |

---

## 9. 完整端到端时序图

```
Client          Proxy                P-Node Scheduler     P-Node Worker        D-Node Scheduler     D-Node Worker
  │               │                       │                    │                      │                    │
  │──POST /v1/completions──▶│              │                    │                      │                    │
  │               │                       │                    │                      │                    │
  │               │──select_prefiller─────│                    │                      │                    │
  │               │──POST max_tokens=1────▶│                   │                      │                    │
  │               │  do_remote_decode=True│                    │                      │                    │
  │               │                       │                    │                      │                    │
  │               │                       │⑭get_num_new_       │                      │                    │
  │               │                       │  matched_tokens()  │                      │                    │
  │               │                       │⑮update_state_      │                      │                    │
  │               │                       │  after_alloc()     │                      │                    │
  │               │                       │⑯build_connector_   │                      │                    │
  │               │                       │  meta()            │                      │                    │
  │               │                       │                    │                      │                    │
  │               │                       │────SchedulerOutput──▶│                     │                    │
  │               │                       │                    │⑳bind_connector_      │                    │
  │               │                       │                    │  metadata()          │                    │
  │               │                       │                    │㉑start_load_kv()     │                    │
  │               │                       │                    │  →add_delayed_request│                    │
  │               │                       │                    │  [模型forward执行]   │                    │
  │               │                       │                    │㉒wait_for_save()→pass│                    │
  │               │                       │                    │㉓get_finished()      │                    │
  │               │                       │                    │㉔clear_connector_    │                    │
  │               │                       │                    │  metadata()          │                    │
  │               │                       │                    │                      │                    │
  │               │                       │⑮request_finished() │                      │                    │
  │               │                       │  →delay_free_blocks │                     │                    │
  │               │                       │  →kv_transfer_params│                     │                    │
  │               │◀─response─────────────│                    │                      │                    │
  │               │  (kv_transfer_params) │                    │                      │                    │
  │               │                       │                    │                      │                    │
  │               │──select_decoder───────│────────────────────│──────────────────────▶│                    │
  │               │──POST + kv_transfer_params────────────────────────────────────────▶│                    │
  │               │  do_remote_prefill=True                    │                      │                    │
  │               │                       │                    │                      │                    │
  │               │                       │                    │                      │④get_num_new_       │
  │               │                       │                    │                      │  matched_tokens()  │
  │               │                       │                    │                      │  →WAITING_FOR_     │
  │               │                       │                    │                      │   REMOTE_KVS       │
  │               │                       │                    │                      │⑤update_state_      │
  │               │                       │                    │                      │  after_alloc()     │
  │               │                       │                    │                      │  →_reqs_need_recv  │
  │               │                       │                    │                      │⑥build_connector_  │
  │               │                       │                    │                      │  meta()            │
  │               │                       │                    │                      │                    │
  │               │                       │                    │                      │──SchedulerOutput──▶│
  │               │                       │                    │                      │                    │⑦bind_connector_
  │               │                       │                    │                      │                    │  metadata()
  │               │                       │                    │                      │                    │⑧start_load_kv()
  │               │                       │                    │                      │                    │  →_get_remote_rank()
  │               │                       │                    │                      │                    │  →add_request()×N
  │               │                       │                    │                      │                    │
  │               │                       │                    │    ┌────ZMQ GET_META_MSG─────────────────▶│
  │               │                       │                    │◀───┤────MooncakeAgentMetadata────────────│
  │               │                       │                    │    │                                     │
  │               │                       │                    │    │  ┌────Mooncake RDMA Read─────────────▶│
  │               │                       │                    │    │  │ (per TP slice, per PP stage)      │
  │               │                       │                    │    │  │ _transfer_kv_cache()              │
  │               │                       │                    │    │  │ →reformat_kv_cache (if MHA)       │
  │               │                       │                    │    │  └───────────────────────────────────│
  │               │                       │                    │    │                                     │
  │               │                       │                    │◀───┤────DONE_RECVING_MSG─────────────────│
  │               │                       │                    │───▶┤────ACK──────────────────────────────│
  │               │                       │                    │    │  (release P-Node blocks)            │
  │               │                       │                    │    │                                     │
  │               │                       │                    │                      │                    │⑫get_finished()
  │               │                       │                    │                      │                    │  →done_recving
  │               │                       │                    │                      │                    │⑬clear_connector_
  │               │                       │                    │                      │                    │  metadata()
  │               │                       │                    │                      │                    │
  │               │                       │                    │                      │⑮update_from_kv_   │
  │               │                       │                    │                      │  xfer_finished()  │
  │               │                       │                    │                      │  →WAITING_FOR_    │
  │               │                       │                    │                      │   REMOTE_KVS      │
  │               │                       │                    │                      │  →WAITING         │
  │               │                       │                    │                      │                    │
  │               │                       │                    │                      │ [正常Decode调度]   │
  │               │◀─────────────────stream response────────────────────────────────────────────────────│
  │               │                       │                    │                      │                    │
```

---

## 10. NPU 特有调用路径差异

与 GPU 路径相比，NPU 的 `model_runner_v1.py` 有以下差异：

| 差异点 | GPU 路径 | NPU 路径 |
|--------|---------|---------|
| KV Connector 封装 | `ActiveKVConnector`（`gpu/kv_connector.py`） | `KVConnectorModelRunnerMixin`（`kv_connector_model_runner_mixin.py`） |
| `register_kv_caches` | `ActiveKVConnector.__init__()` 中调用 | `model_runner_v1.py:L2788-2789` 直接调用 `get_kv_transfer_group().register_kv_caches()` |
| Forward Context | `set_forward_context()` | `set_ascend_forward_context()`（含 NPU 特有参数） |
| `clear_connector_metadata` | `ActiveKVConnector.post_forward()` 中调用 | `model_runner_v1.py:L1721-1722` 直接调用 `get_kv_transfer_group().clear_connector_metadata()` |
| KV Cache 格式 | 标准 paged attention | 支持 NZ 格式（`enable_kv_nz`），融合 reformat 算子 |
