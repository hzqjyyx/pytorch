## MetaTensor.cpp 文件分析

这个文件实现了两个核心的元张量（Meta Tensor）操作函数：

### 主要功能

**1. `empty_meta_symint` 函数（第14-29行）**
- 创建一个空的元张量，支持符号化整数尺寸
- 首先尝试将符号化尺寸转换为常规 IntArrayRef
- 如果转换成功，调用 `at::detail::empty_meta` 处理固定大小的情况
- 如果包含符号化尺寸，调用 `at::detail::empty_symint_meta` 处理动态大小的情况
- 支持指定数据类型、内存布局、设备、是否锁页内存和内存格式

**2. `empty_strided_meta_symint` 函数（第31-41行）**
- 创建指定步幅（stride）的空元张量
- 接受自定义的尺寸和步幅参数（都支持符号化整数）
- 直接委托给 `at::detail::empty_strided_symint_meta` 实现具体逻辑
- 支持数据类型、内存布局和设备的自定义

### 核心特点

- **符号化整数支持**：通过 `SymIntArrayRef` 处理动态形状，用于 TorchScript 和 AOT 编译等场景
- **延迟分配**：元张量仅记录张量的元数据（形状、步幅等），不分配实际内存
- **灵活配置**：支持多种张量属性的自定义设置

### 快速总结

• 实现两个元张量创建函数，支持符号化尺寸
• `empty_meta_symint`：通用空张量创建，智能处理固定/动态尺寸
• `empty_strided_meta_symint`：支持自定义步幅的空张量创建
• 用于 PyTorch 的 meta 执行模式，仅操作张量元数据而不分配内存
