## TensorMeta.h 和 TensorMeta.cpp 文件分析

**TensorMeta.h** 定义了 PyTorch 中的元函数（meta function）框架，这是结构化内核（structured kernels）的核心组成部分。

### 核心概念

**MetaBase 虚基类**（第 66-133 行）
- 所有结构化内核类的基类
- 定义了设置输出张量属性的虚方法接口
- 子类包括 TensorIteratorBase

**宏定义系统**

1. **元函数定义宏**（第 27-39 行）
   - `TORCH_META_FUNC(name)` - 单参数元函数原型
   - `TORCH_META_FUNC2(name, overload)` - 带重载名的元函数原型
   - 预计算版本：`TORCH_PRECOMPUTE_META_FUNC` 和 `TORCH_PRECOMPUTE_META_FUNC2`
   - 用途：在编译时生成 `structured_<name>::meta` 成员函数

2. **实现函数定义宏**（第 58 行）
   - `TORCH_IMPL_FUNC(name)` - 生成 `structured_<name>::impl` 成员函数

3. **预计算结构体宏**（第 42-44 行）
   - 用于在元函数中返回预计算的值

### set_output 方法决策树（第 74-90 行）

根据内核对输出张量步幅（strides）的需求，选择合适的方法：

- **set_output_raw_strided** - 支持任意步幅
- **set_output_strided** - 要求特定步幅，不匹配时创建代理输出
- **set_output_contiguous** - 要求连续步幅（set_output_strided 的包装）

### 主要功能

- 定义元函数的原型和签名约定
- 提供虚接口让子类实现张量输出属性的设置
- 通过宏系统简化结构化内核的定义和实现
- 支持预计算值在元函数和实现函数之间传递

**TensorMeta.cpp** 为空实现文件，仅包含命名空间声明。

---

### 核心功能清单

- **元函数框架** - 为 PyTorch 结构化内核定义元计算（shape/dtype 推导）的接口
- **宏封装系统** - 简化内核定义，自动生成 `structured_<name>::meta/impl` 成员函数
- **输出张量管理** - 通过 `MetaBase` 虚方法控制输出张量的大小、步幅和选项
- **步幅策略** - 提供三层方法供内核选择（任意/特定/连续）
- **预计算支持** - 允许元函数返回预计算结构体给实现函数使用
