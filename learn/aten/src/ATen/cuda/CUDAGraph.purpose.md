# CUDAGraph 核心功能分析

## 主要目的

这是 PyTorch 对 CUDA Graph API 的封装类，用于捕获和重放 CUDA 操作序列，以减少 kernel 启动开销和提高性能。

## 核心机制

### 1. 图捕获流程 (capture_begin → capture_end)

**capture_begin** (aten/src/ATen/cuda/CUDAGraph.cpp:58-116):
- 检查当前流必须是非默认流
- 注册默认随机数生成器并调用所有生成器的 `capture_prologue()`
- 创建或共享内存池 (mempool_id_)，用于图的内存分配
- 调用 `beginAllocateToPool()` 让 caching allocator 将后续分配导向该内存池
- 调用 `cudaStreamBeginCapture()` 开始捕获，获取 capture_id_

**capture_end** (aten/src/ATen/cuda/CUDAGraph.cpp:118-188):
- 必须在同一个流上结束捕获
- 调用 `cudaStreamEndCapture()` 获得 graph_
- 调用 `endAllocateToPool()` 结束内存池分配
- 根据 CUDA 版本选择实例化方式：
  - CUDA < 11.4: 使用 `cudaGraphInstantiate()`
  - CUDA >= 11.4: 使用 `cudaGraphInstantiateWithFlags()` 并设置 `cudaGraphInstantiateFlagAutoFreeOnLaunch`
- 调用所有生成器的 `capture_epilogue()` 并保存 wholegraph_increments
- 检查图节点数，为空则发出警告
- 非调试模式下立即销毁 graph_，只保留 graph_exec_

### 2. 图重放 (replay)

**replay** (aten/src/ATen/cuda/CUDAGraph.cpp:190-212):
- 调用所有生成器的 `replay_prologue(wholegraph_increments)` 恢复 RNG 状态
- 调用 `cudaGraphLaunch()` 在当前流上执行图
- CUDA < 11.4 需要同步以规避 libcuda.so 的 bug

### 3. 内存池管理

三种内存池创建方式：

1. **默认模式** (pool={0,0}): `mempool_id_.first > 0`, `.second = 0`
2. **共享其他图的池** (pool=other_graph.pool()): 复用 mempool_id_
3. **共享通过 graph_pool_handle() 创建的池**: `mempool_id_.first = 0`, `.second > 0`

内存池在 capture_begin 时创建/关联，在 reset/析构时通过 `releasePool()` 释放。

### 4. 随机数生成器协作

**问题背景** (Note [CUDA Graph-safe RNG states]):
- CUDA 图重放时必须保证数值一致性
- PyTorch 的某些 RNG 操作有 CPU 状态依赖

**解决方案**:
- 捕获阶段：`capture_prologue()` 准备状态，`capture_epilogue()` 保存增量
- 重放阶段：`replay_prologue(wholegraph_increments)` 恢复正确的 RNG 状态
- 析构时：`unregister_graph()` 清理关联

### 5. 资源清理

**reset** (aten/src/ATen/cuda/CUDAGraph.cpp:236-268):
- 释放内存池
- 销毁 graph_ (如果存在)
- 销毁 graph_exec_ (如果存在)
- 使用 `C10_CUDA_CHECK_WARN` 而非抛异常（因为会在析构函数中调用）

**析构函数** (aten/src/ATen/cuda/CUDAGraph.cpp:277-295):
- 遍历所有生成器并调用 `unregister_graph()`
- 调用 reset() 清理资源

## 关键设计考量

### 为什么在 PyTorch 核心而非用户扩展？

1. **便利性**: 提供统一的捕获/重放接口
2. **正确性**: 确保与 native CUDA ops (尤其是 RNG) 的正确交互

### cudaGraphInstantiateFlagAutoFreeOnLaunch (CUDA 11.4+)

使用场景：当使用 cudaMallocAsync 后端且某些张量在重放之间不释放时，需要此标志让图在启动后自动释放临时内存。

### 调试支持

- `enable_debug_mode()`: 设置 `_cuda_graphs_debug = true`
- `debug_dump(path)`: 在 CUDA 11.3+ 时调用 `cudaGraphDebugDotPrint()` 输出详细图结构
- 调试模式下 capture_end 不会销毁 graph_，保留到 debug_dump 调用

## 主要数据成员

- `graph_`: 捕获的 CUDA 图对象
- `graph_exec_`: 实例化后的可执行图
- `capture_id_`: CUDA 分配的捕获 ID，用于识别参与捕获的流
- `mempool_id_`: 内存池标识符，支持池共享
- `capture_stream_`: 捕获开始的流
- `capture_dev_`: 捕获发生的设备
- `captured_generator_states_`: 管理的生成器状态及其增量映射

## 使用限制

- 必须在非默认流上捕获（但可在默认流上重放）
- 同一 CUDAGraph 实例只能捕获一次（除非 reset）
- 捕获必须在同一流上开始和结束
- 当前实现要求所有操作在同一设备上（可扩展但尚未实现）

---

### ROCm 相关差异
- ROCm 6.2+ 支持 `cudaGraphInstantiateFlagAutoFreeOnLaunch`
- ROCm 6.2+ 的 `hipGraphExecDestroy` 延迟释放内存，需在析构时同步

### Backward 相关
- 生成器管理中的 `wholegraph_increments` 用于反向传播时正确恢复 RNG 状态
