## DeviceGuard.h 主要功能

这个文件定义了两个 RAII 类用于管理 PyTorch 中的设备上下文：

### DeviceGuard 类
- 在构造函数中设置指定的设备为当前设备
- 在析构函数中自动恢复到构造时的原始设备
- 不允许默认构造、拷贝或移动，确保始终处于初始化状态
- 提供 `reset_device()` 和 `set_index()` 方法更改当前设备
- 提供 `original_device()` 和 `current_device()` 方法查询设备状态

### OptionalDeviceGuard 类
- 功能类似 DeviceGuard，但可以处于未初始化状态
- 支持默认构造（创建未初始化的守卫）
- 支持 `std::optional<Device>` 初始化
- `original_device()` 和 `current_device()` 返回 `std::optional<Device>`
- 用于可选地应用设备守卫或循环处理多个张量时避免重复重置

### 核心特点

- **RAII 模式**：自动管理设备切换和恢复
- **类型安全**：使用 `InlineDeviceGuard` 和 `VirtualGuardImpl` 提供编译时优化
- **防错设计**：不允许移动和拷贝，避免资源泄漏
- **测试支持**：提供额外的构造函数接收 `DeviceGuardImplInterface` 用于单元测试

### 主要用途

- **自动设备管理**：确保在特定作用域内使用指定的计算设备
- **异常安全**：即使发生异常也能正确恢复设备状态
- **性能优化**：OptionalDeviceGuard 在循环中可避免每次迭代都重置设备
