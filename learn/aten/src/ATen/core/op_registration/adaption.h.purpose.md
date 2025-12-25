这个文件定义了操作注册(op registration)中的适配工具函数，主要用于在调度器(dispatcher)和内核实现之间进行参数转换。

## 核心功能

**1. 可选张量的包装移除 [hacky wrapper removal for optional tensor]**
- 问题：内核实现接收模式中标记的 `Tensor?`（可选张量），但 C++ 函数期望 `std::optional<Tensor>`
- 解决方案：C++ 函数改为接收 `std::optional<Tensor>`，在函数开始处使用 `at::borrow_from_optional_tensor()` 解包
- 示例：`c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);`

**2. TensorOptions 的包装移除 [hacky wrapper removal for TensorOptions]**
- 问题：内核实现接收 `TensorOptions` 参数，但调度器期望 4 个独立参数（dtype、layout、device、pin_memory）
- 解决方案：内核改为接收这 4 个分离参数，然后在函数开始处组装成 `TensorOptions`
- 示例：`TensorOptions options = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);`

**3. 设备一致性检查 [check_and_update_common_device]**
- 一组重载函数用于验证多个张量是否在同一设备上
- 支持多种输入类型：单个张量、可选张量、张量列表、可选张量列表
- 如果张量在不同设备上会触发 `common_device_check_failure()`

## 主要内容

- **命名空间**：`c10::impl`
- **公共函数**：`common_device_check_failure()` - 设备检查失败处理器
- **内联函数**：4 个重载的 `check_and_update_common_device()` 函数

---

**关键特点：**
- 提供从调度器参数格式到内核参数格式的转换适配
- 确保多张量操作中的设备一致性
- 设计用于简化内核实现与调度器交互的不匹配问题
