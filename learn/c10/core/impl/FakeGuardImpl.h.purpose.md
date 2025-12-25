这个文件定义了一个用于测试的假设备保护实现（`FakeGuardImpl`）。它模拟了 CUDA/GPU 设备管理的行为，而不需要实际的硬硬件。

**核心结构：**

`FakeGuardImpl` 是一个模板类，继承自 `DeviceGuardImplInterface`，它：

1. **设备管理** - 维护当前设备索引（`current_device_`），支持最多 8 个假设备
2. **设备切换** - 通过 `exchangeDevice()` 和 `setDevice()` 方法切换当前设备
3. **流管理** - 为每个设备维护一个流 ID（`current_streams_`），支持获取和交换流
4. **设备数查询** - `deviceCount()` 返回固定的 8 个设备
5. **事件操作** - 实现事件相关接口（`record`、`block`、`queryEvent`、`destroyEvent`），但大多数为空实现或直接返回成功

**关键特性：**

- **线程本地存储** - 使用 `thread_local` 静态变量确保每个线程有独立的设备和流状态
- **测试便利性** - 提供静态方法（`getDeviceIndex`、`setDeviceIndex`、`resetStreams`）供测试代码直接操作
- **类型安全** - 通过模板参数 `T` 指定设备类型，运行时进行类型检查

**主要用途：**

- 单元测试中模拟 GPU 设备管理行为
- 验证设备保护机制（DeviceGuard）的逻辑，无需实际 GPU

**关键方法列表：**

- `getDevice()` / `setDevice()` - 获取/设置当前设备
- `exchangeDevice()` - 原子地切换设备并返回旧设备
- `getStream()` / `exchangeStream()` - 流管理
- `deviceCount()` - 返回 8
- `record()` / `block()` / `queryEvent()` - 事件操作（空实现或无操作）
- `resetStreams()` - 重置所有流为 0
