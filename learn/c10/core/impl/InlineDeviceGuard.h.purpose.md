# InlineDeviceGuard.h 核心功能

这个文件提供了两个 RAII 类模板的实现，用于自动管理设备切换：

## InlineDeviceGuard<T>

**设计目的**：在构造时切换到指定设备，析构时自动恢复到原设备。

**模板参数 T**：
- 具体的设备实现（如 `CUDAGuardImpl`）：完全去虚拟化，性能最优
- `VirtualGuardImpl`：通过虚拟分发支持运行时设备类型选择

**核心机制**：
```cpp
// 构造时：
// 1. 如果 device.index() == -1，只获取当前设备，不切换
// 2. 否则通过 exchangeDevice() 切换到新设备并记录原设备
original_device_ = device.index() == -1 ? impl_.getDevice() 
                                        : impl_.exchangeDevice(device)

// 析构时：
~InlineDeviceGuard() {
    impl_.uncheckedSetDevice(original_device_);  // 无条件恢复
}
```

**关键方法**：
- `set_device(device)`：切换到新设备，更新 `current_device_`
- `reset_device(device)`：
  - 非虚拟版本：等同于 `set_device`
  - 虚拟版本：支持跨设备类型切换，会先恢复原设备，再重新初始化 impl_
- `set_index(index)`：保持设备类型，只切换索引

**设计限制**：
- 删除默认构造函数：避免无意义的"恢复到当前设备"操作
- 删除拷贝/移动：无未初始化状态，移动语义不适用

## InlineOptionalDeviceGuard<T>

**与 InlineDeviceGuard 的区别**：允许未初始化状态。

**实现**：
```cpp
std::optional<InlineDeviceGuard<T>> guard_;
```

**使用场景**：
- 处理可能为空的设备（如未定义的 Tensor）
- 延迟初始化（构造时不切换，稍后调用 `set_device` 时才初始化）

**关键特性**：
- 空构造函数创建未初始化 guard
- `std::optional<Device>` 构造函数：仅当有值时初始化内部 guard
- `set_device/reset_device`：首次调用时 emplace 初始化，后续调用委托给内部 guard
- `reset()` 方法：销毁内部 guard，恢复原设备，回到未初始化状态

**同样禁止拷贝/移动**：
- 拷贝：不符合 RAII 语义
- 移动构造：多层嵌套时原设备恢复顺序不明确
- 移动赋值：无法确定哪个 guard 的 `original_device_` 应该生效

---

**其他提及但忽略的内容**：
- ROCm/HIP 设备类型兼容性检查（line 122）
- nvcc 编译器 bug 的 workaround（Note [Explicit initialization of optional fields]）
