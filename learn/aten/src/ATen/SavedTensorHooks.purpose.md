## SavedTensorHooks 主要功能

这两个文件实现了 PyTorch 中保存张量钩子（Saved Tensor Hooks）的默认管理机制，用于在反向传播过程中控制张量的打包和解包行为。

### 核心结构

**SavedTensorDefaultHooksTLS** (线程本地存储)：
- `stack`：存储成对的打包/解包钩子（SafePyObject 对）
- `disabled_error_message`：禁用状态标记（nullopt 表示启用，有值表示禁用并附带错误消息）
- `is_tracing`：追踪模式标志，用于延迟钩子执行

### 主要操作

1. **钩子栈管理**
   - `push_hooks()` - 将新的钩子对压入栈
   - `pop_hooks()` - 从栈中弹出钩子对
   - `get_hooks()` - 获取当前栈顶的钩子对（如果存在且未在追踪）

2. **启用/禁用控制**
   - `disable(message)` - 禁用钩子，如果栈非空则抛出错误
   - `enable()` - 重新启用钩子
   - `is_enabled()` - 检查钩子是否启用
   - `get_disabled_error_message()` - 获取禁用原因

3. **状态管理**
   - `get_tls_state()` / `set_tls_state()` - 保存/恢复线程本地状态
   - `lazy_initialize()` - 首次初始化标记
   - `set_tracing()` - 切换追踪模式

### 关键特性

- **线程隔离**：使用 `thread_local` 存储，每个线程独立管理
- **延迟初始化**：通过 `is_initialized` 标志优化未使用钩子的场景
- **追踪模式**：Dynamo/AOTAutograd 在追踪时延迟钩子执行以保留急切执行语义
- **错误报告**：禁用时提供明确的错误消息

### 使用场景

• 张量保存时打包处理（如压缩存储）  
• 反向传播时张量解包还原  
• 动态编译/追踪过程中的钩子控制  
• 不兼容功能的临时禁用机制  
• 跨线程张量钩子状态隔离
