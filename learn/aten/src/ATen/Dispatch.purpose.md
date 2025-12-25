## 核心功能

这两个文件实现了 PyTorch 的**类型分发（Type Dispatch）系统**，允许针对不同数据类型（dtype）生成专门化的内核代码。

### Dispatch.h - 宏系统核心

这是一个纯头文件，提供了一系列宏来实现编译期类型分发：

**基础机制：**
- `AT_DISPATCH_SWITCH` 是底层宏，生成一个 switch-case 语句，根据运行时的 `ScalarType` 枚举值跳转到对应分支
- 每个 case 分支内部定义类型别名 `scalar_t`（通过 `ScalarTypeToCPPTypeT` 映射，如 `ScalarType::Float` → `float`）
- 执行用户提供的 lambda 表达式，lambda 内可使用 `scalar_t` 编写泛型代码

**选择性编译支持：**
- `should_include_kernel_dtype()` - 判断某个 dtype 是否应该被编译（用于移动端精简构建）
- `AT_PRIVATE_CHECK_SELECTIVE_BUILD` - 运行时检查，如果 dtype 未被选中则抛出错误
- `RECORD_KERNEL_FUNCTION_DTYPE` - 在 Facebook 内部构建时记录内核调用，用于追踪模型执行以生成精简配置

**分发宏族：**

按数据类型分组提供不同的宏变体：

1. **浮点类型**
   - `AT_DISPATCH_FLOATING_TYPES`: Double, Float
   - `AT_DISPATCH_FLOATING_TYPES_AND_HALF`: 加上 Half (FP16)
   - `AT_DISPATCH_REDUCED_FLOATING_TYPES`: 只有 Half, BFloat16

2. **整数类型**
   - `AT_DISPATCH_INTEGRAL_TYPES`: Byte(uint8), Char(int8), Short(int16), Int(int32), Long(int64)

3. **复数类型**
   - `AT_DISPATCH_COMPLEX_TYPES`: ComplexFloat, ComplexDouble

4. **组合类型**
   - `AT_DISPATCH_ALL_TYPES`: 浮点 + 整数
   - `AT_DISPATCH_ALL_TYPES_AND_COMPLEX`: 浮点 + 整数 + 复数
   - `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES`: 浮点 + 复数

5. **扩展宏**
   - `AT_DISPATCH_*_AND(SCALARTYPE, ...)`: 在基础类型集合上额外添加一个类型
   - `AT_DISPATCH_*_AND2/3/4/5/6/7/8`: 添加多个额外类型

6. **量化类型**
   - `AT_DISPATCH_QINT_TYPES`: qint8, quint8, qint32
   - `AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES`: 额外包含 quint4x2, quint2x4（sub-byte 类型）

7. **其他**
   - `AT_DISPATCH_BIT_TYPES`: Bits1x8, Bits2x4, Bits4x2, Bits8, Bits16
   - `AT_DISPATCH_INDEX_TYPES`: Int, Long（用于索引操作）

**Switch-Case 语法：**

除了一体化的宏，还提供了更灵活的开关语法：
```cpp
AT_DISPATCH_SWITCH(dtype, "op_name",
    AT_DISPATCH_CASE_INTEGRAL_TYPES([&] { /* 整数实现 */ })
    AT_DISPATCH_CASE_FLOATING_TYPES([&] { /* 浮点实现 */ })
    AT_DISPATCH_CASE(kBool, [&] { /* bool 专门实现 */ })
)
```

### Dispatch.cpp - 运行时追踪

仅在 `ENABLE_RECORD_KERNEL_FUNCTION_DTYPE` 宏启用时编译：

- `record_kernel_function_dtype()` - 使用 `RECORD_FUNCTION_WITH_SCOPE` 记录内核函数名和 dtype
- 用于 Facebook 内部构建中的追踪二进制，通过记录模型执行时使用的 dtype 组合，生成精简构建配置

### 典型使用模式

```cpp
// 用户代码
AT_DISPATCH_ALL_TYPES(tensor.scalar_type(), "my_kernel", [&] {
    // scalar_t 现在是具体的 C++ 类型（float/double/int32_t等）
    my_kernel_impl<scalar_t>(tensor.data_ptr<scalar_t>());
});
```

展开后等价于：
```cpp
switch (tensor.scalar_type()) {
    case ScalarType::Float: {
        using scalar_t = float;
        my_kernel_impl<float>(tensor.data_ptr<float>());
        break;
    }
    case ScalarType::Double: { /* ... */ }
    // ... 其他类型
}
```

### 设计目标

- **避免代码重复**：一份模板代码自动生成多个类型特化版本
- **编译期优化**：每个分支都是单态化的，编译器可以充分优化
- **构建大小控制**：通过选择性编译减少移动端二进制体积
- **类型安全**：编译期保证类型匹配，避免运行时类型转换错误

---

### 简要列出的内容

**ROCm 相关：**
- 文件中没有直接的 ROCm 特定代码，但宏系统被 CUDA/ROCm 内核使用

**Backward 相关：**
- 这是前向/后向无关的底层基础设施，自动微分系统会使用这些宏来分发梯度计算内核
