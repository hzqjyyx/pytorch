# CUDAGeneratorImpl 核心功能

## 整体职责
实现 PyTorch 的 CUDA 随机数生成器，管理 GPU 上的随机数生成状态，支持普通执行和 CUDA Graph 捕获/重放两种模式。

## 核心组件

### CUDAGeneratorState
**状态管理器**，包含：
- `seed_`: 随机数种子
- `philox_offset_per_thread_`: 普通模式下的 Philox 偏移量
- `offset_intragraph_`: Graph 内部的相对偏移量
- `capturing_`: 标记是否处于 Graph 捕获模式
- `seed_extragraph_` / `offset_extragraph_`: GPU 设备上的张量，用于 Graph 重放时存储初始状态

**关键方法**：
- `increase(increment)`: 根据当前模式（捕获/普通）增加对应的偏移量，确保偏移量是 4 的倍数
- `register_graph()` / `unregister_graph()`: 注册/注销 CUDA Graph，首次注册时分配 GPU 张量
- `capture_prologue()`: Graph 捕获前重置状态，将当前 seed 写入 GPU 张量
- `capture_epilogue()`: 捕获结束，返回累计的 graph 内偏移量
- `replay_prologue()`: Graph 重放前准备，将当前状态写入 GPU 张量

### CUDAGeneratorImpl
**生成器实现类**，继承自 `c10::GeneratorImpl`

**构造与初始化**：
- `initCUDAGenVector()`: 全局初始化，获取 GPU 数量，准备默认生成器容器
- `getDefaultCUDAGenerator()`: 每个 GPU 设备一个默认生成器，延迟初始化
- `createCUDAGenerator()`: 创建新的独立生成器实例

**状态操作**：
- `set_current_seed()` / `current_seed()`: 设置/获取种子
- `set_offset()` / `get_offset()`: 设置/获取偏移量
- `seed()`: 从 `/dev/urandom` 获取非确定性随机数作为种子
- `get_state()` / `set_state()`: 序列化/反序列化状态为 CPU 字节张量
- `graphsafe_set_state()` / `graphsafe_get_state()`: 切换内部状态指针，用于多状态管理

**核心随机数生成接口**：
```cpp
PhiloxCudaState philox_cuda_state(uint64_t increment)
```
**双模式设计**：
- **普通模式**：返回包含 `(seed, offset)` 的状态，然后增加 `philox_offset_per_thread_`
- **Graph 捕获模式**：返回包含 GPU 张量指针 `(seed_extragraph_, offset_extragraph_)` 和当前 `offset_intragraph_`，然后增加 `offset_intragraph_`

消费端 kernel 使用 `at::cuda::philox::unpack()` 解包，自动处理两种模式的差异。

## CUDA Graph 支持机制

### 设计思路
Graph 内的所有 RNG 操作对外表现为一个整体：
- **捕获时**：记录 graph 内累计的偏移量增量
- **重放时**：
  1. 将当前生成器的 seed 和 offset 写入 GPU 张量
  2. Kernel 读取 GPU 张量值作为基准偏移量
  3. 加上 kernel 的 intra-graph 偏移量得到最终偏移量
  4. 重放后增加生成器的全局偏移量

### 工作流程
```
Capture:
  capture_prologue() → reset offset_intragraph_ = 0
  → kernel calls philox_cuda_state() → 返回 GPU 张量指针 + intra-graph offset
  → capture_epilogue() → 返回总增量

Replay:
  replay_prologue(total_increment) 
  → 写入当前 seed/offset 到 GPU 张量
  → increase(total_increment) 更新全局偏移量
  → kernel 读取 GPU 张量 + intra-graph offset
```

### 显式注册机制
用户必须显式调用 `register_graph()`，因为在捕获阶段重置 GPU 张量会被录入 graph，导致重放时错误重置。显式注册确保 `capture_prologue()` 在捕获开始前完成。

## 技术约束

### 偏移量对齐要求
所有偏移量必须是 4 的倍数：
- Philox 算法将 offset 除以 4 定位到 128-bit 块
- 每个线程依次返回 4 个 32-bit 随机数
- 非对齐偏移会导致随机数重复使用

### 线程安全
大部分操作需要在调用端使用 `std::lock_guard<std::mutex> lock(gen->mutex_)` 保护。

### Graph 捕获限制
许多方法在 Graph 捕获期间禁止调用（通过 `assertNotCapturing()` 检查）：
- `set_current_seed()`, `set_offset()`, `seed()`
- `get_offset()`, `current_seed()`
- `clone_impl()`, `set_state()`

---

**ROCm 相关**：
- 无明显 ROCm 特定代码，但通过 `c10::cuda` 抽象层兼容 ROCm

**Backward 相关**：
- 无反向传播逻辑，RNG 状态不参与自动微分
