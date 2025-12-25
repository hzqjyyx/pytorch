# ScalarType 核心功能理解

## 类型系统定义

这两个文件定义了 PyTorch 的标量类型系统，是整个框架的类型基础设施。

**ScalarType.h** 通过枚举定义了所有支持的数据类型：
- 整数类型：`Byte(uint8)`, `Char(int8)`, `Short(int16)`, `Int(int32)`, `Long(int64)`
- 浮点类型：`Half(fp16)`, `Float(fp32)`, `Double(fp64)`, `BFloat16`
- 复数类型：`ComplexHalf`, `ComplexFloat`, `ComplexDouble`
- 量化类型：`QInt8`, `QUInt8`, `QInt32`, `QUInt4x2`, `QUInt2x4`
- Float8 变体：`Float8_e5m2`, `Float8_e4m3fn`, `Float8_e5m2fnuz`, `Float8_e4m3fnuz`, `Float8_e8m0fnu`
- Bits 类型：`Bits1x8`, `Bits2x4`, `Bits4x2`, `Bits8`, `Bits16`
- 扩展无符号整数：`UInt16`, `UInt32`, `UInt64`, `UInt1-7`（dummy 实现）
- 扩展有符号整数：`Int1-7`（dummy 实现）
- 布尔类型：`Bool`

## C++ 类型映射

**双向映射机制**：
- `ScalarTypeToCPPType<N>`：从 ScalarType 枚举映射到 C++ 类型（如 `ScalarType::Float -> float`）
- `CppTypeToScalarType<T>`：从 C++ 类型映射回 ScalarType（如 `float -> ScalarType::Float`）

这些映射通过宏 `AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS` 自动生成特化模板。

## 类型提升规则

**ScalarType.cpp** 中的 `promoteTypes()` 实现了类似 NumPy 的类型提升逻辑：

```cpp
ScalarType promoteTypes(ScalarType a, ScalarType b)
```

关键规则：
1. **相同类型直接返回**：`a == b` 时返回该类型
2. **特殊类型处理**：
   - 量化类型（QInt）：抛出异常，未实现
   - Float8 类型：不支持提升，抛出异常
   - Bits 类型：返回 `Undefined`
   - 无符号类型（uint16/32/64）：仅支持浮点提升，否则抛出异常

3. **查表提升**：通过 13x13 的 `_promoteTypesLookup` 静态表查找结果
   - 行列索引对应两个输入类型
   - 表格按固定顺序：`u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, bf`
   - 示例：`promoteTypes(int8, float16) -> float16`

**索引映射**：
- `index2dtype`：从索引映射到 ScalarType（编译期常量数组）
- `dtype2index`：从 ScalarType 映射到索引（通过 `calculate_dtype2index()` 在编译期计算）

## 类型查询工具函数

**分类判断**：
- `isIntegralType(t, includeBool)`：整数类型（含 uint16/32/64）
- `isFloatingType(t)`：浮点类型（包括 half/bfloat16/float8）
- `isComplexType(t)`：复数类型
- `isQIntType(t)`：量化类型
- `isBitsType(t)`：Bits 类型
- `isFloat8Type(t)`：Float8 变体
- `isReducedFloatingType(t)`：降精度浮点（half/bfloat16/float8）
- `isBarebonesUnsignedType(t)`：扩展无符号类型
- `isSignedType(t)`：有符号类型（基于 `std::numeric_limits`）

**类型转换**：
- `toQIntType(t)`：普通整数转量化类型
- `toUnderlying(t)`：量化类型转底层存储类型
- `toRealValueType(t)`：复数转实数分量类型
- `toComplexType(t)`：实数转对应复数类型
- `canCast(from, to)`：判断类型转换是否合法
  - 禁止：复数→非复数、浮点→整数、非布尔→布尔

## 字符串转换

**`getDtypeNames()`**：返回 `(标准名, 遗留名)` 对
- 示例：`Float -> ("float32", "float")`、`Byte -> ("uint8", "")`

**`getStringToDtypeMap()`**：返回全局字符串到 ScalarType 的映射表
- 懒加载单例模式
- 同时注册标准名和遗留名

**`toString()`**：ScalarType 转字符串（返回枚举名）

**`elementSize()`**：返回类型字节大小（通过 `sizeof`）

## 宏系统

定义了多组宏用于批量操作不同类型集合：
- `AT_FORALL_SCALAR_TYPES`：基础类型（整数+浮点）
- `AT_FORALL_SCALAR_TYPES_WITH_COMPLEX`：含复数的常用类型
- `AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS`：全量类型（含量化）
- `AT_FORALL_INT_TYPES`：仅整数类型
- `AT_FORALL_QINT_TYPES`：仅量化类型
- `AT_FORALL_FLOAT8_TYPES`：仅 Float8 类型
- `AT_FORALL_COMPLEX_TYPES`：仅复数类型
- `AT_FORALL_SCALAR_TYPES_AND[2/3/7]`：基础类型+额外指定类型

这些宏广泛用于模板实例化控制和内核分发。

---

**额外内容**（ROCm/Backward 相关）：
- 无 ROCm 特定逻辑
- 无反向传播相关代码
- 文件属于纯前向声明和工具函数层
