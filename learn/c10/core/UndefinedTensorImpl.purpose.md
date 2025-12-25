## UndefinedTensorImpl 概述

`UndefinedTensorImpl` 是 PyTorch 中用于表示**未定义张量**的特殊张量实现。它是一个单例模式的特殊 TensorImpl，用于表示那些没有实际数据或元数据的张量状态。

### 核心设计

**单例模式实现**：
- Windows：使用 `getInstance()` 静态方法返回单例实例
- 其他平台：使用静态成员变量 `_singleton` 作为全局实例

**构造函数**（私有）：
- 使用 `DispatchKey::Undefined` 标记为未定义张量类型
- 设置 `storage_access_should_throw()` - 任何存储访问都会抛出异常
- 使用自定义的 Strides 策略

### 主要特性

**禁止操作**（调用时抛出异常）：
- `strides_custom()` - 获取步长
- `sym_strides_custom()` - 获取符号步长
- `set_storage_offset()` - 设置存储偏移
- 任何存储访问操作

**允许操作**：
- `is_contiguous_custom()` - 检查连续性（代理到默认实现）
- `has_storage()` - 在 DEBUG 模式下验证存储为空

**元数据**：
- 类型名称：`"UndefinedTensorImpl"`
- 禁止访问大小/步长元数据（虽然当前未强制）

### 应用场景

表示张量操作返回"未定义"结果的情况，防止误用未初始化的张量。

---

**功能总结：**
- 单例模式的特殊张量实现，表示未定义状态
- 所有有意义的操作（strides、storage、offset）都会显式抛出异常
- 通过 DispatchKey::Undefined 进行类型标记
- 设计用于捕获对未定义张量的非法操作
