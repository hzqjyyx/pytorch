## Scalar 类的核心功能

`c10::Scalar` 是 PyTorch 中表示**单个标量值**的类型，可以看作是 0 维张量。它的设计目标是提供一个统一的接口来处理不同类型的数值，并且支持从 C++ 字面量到 Scalar 的隐式转换。

### 存储结构

使用 **tagged union** 模式存储值：

```cpp
enum class Tag { HAS_d, HAS_i, HAS_u, HAS_z, HAS_b, HAS_sd, HAS_si, HAS_sb };
union v_t {
    double d;           // 浮点数
    int64_t i;          // 有符号整数
    uint64_t u;         // 无符号整数（仅当无法用 int64_t 表示时）
    complex<double> z;  // 复数
    intrusive_ptr_target* p;  // 符号类型的指针
};
```

- `tag` 字段标识当前存储的值类型
- `HAS_u` 仅用于存储无法用 `int64_t` 表示的 `uint64_t`（大于 INT64_MAX 的值）
- `HAS_sd/HAS_si/HAS_sb` 用于存储符号类型（SymFloat/SymInt/SymBool）的指针

### 类型系统

支持的核心类型分类：

1. **浮点类型** (`isFloatingPoint`): `HAS_d` 和 `HAS_sd`
2. **整数类型** (`isIntegral`): `HAS_i`, `HAS_si`, `HAS_u`（可选包含布尔值）
3. **复数类型** (`isComplex`): `HAS_z`
4. **布尔类型** (`isBoolean`): `HAS_b` 和 `HAS_sb`
5. **符号类型** (`isSymbolic`): `HAS_si`, `HAS_sd`, `HAS_sb`

### 构造与转换

**隐式构造**：
- 支持所有标量类型的隐式构造（int, float, double, complex, Half, BFloat16 等）
- 特殊处理 `bool` 类型以避免与 `Value*` 的歧义
- 对 `long`/`long long` 类型在不同平台上有特殊适配

**类型转换**（`to*` 方法）：
- 提供 `toInt()`, `toDouble()`, `toFloat()` 等转换方法
- 使用 `checked_convert` 进行安全的类型转换，防止溢出
- 对符号类型会调用 `guard_int`/`guard_float`/`guard_bool` 具体化

### 数学运算

在 Scalar.cpp 中实现了三个基本运算：

1. **取负** (`operator-`): 
   - 不支持布尔类型
   - 对浮点、复数、整数分别处理
   - 符号类型尚未实现（NYI）

2. **共轭** (`conj`):
   - 仅对复数有意义（`std::conj`）
   - 其他类型返回自身

3. **对数** (`log`):
   - 支持复数、浮点、整数
   - 符号类型尚未实现

### 资源管理

- 符号类型（`HAS_si/sd/sb`）使用侵入式指针管理生命周期
- `destroy()` 在析构时释放符号类型的引用计数
- 拷贝构造增加引用计数，移动构造转移所有权
- 移动后的源对象被重置为 `HAS_i` 标签且值为 0

### 相等比较

`equal()` 方法支持：
- 与非复数类型比较（处理溢出检查）
- 与复数类型比较（实部虚部分别比较）
- 布尔值比较（仅当 Scalar 本身是布尔类型）
- 符号类型的相等比较尚未实现

### 类型查询

`type()` 方法返回 `ScalarType` 枚举：
- 复数 → `ComplexDouble`
- 浮点 → `Double`
- 有符号整数 → `Long`
- 无符号整数 → `UInt64`（仅当 `HAS_u`）
- 布尔 → `Bool`

---

**相关要点：**
• 符号类型（SymInt/SymFloat/SymBool）支持符号形状推理，但部分运算标记为 NYI
• 平台特定处理：macOS/Windows/Linux 对 long/long long 的类型定义不同
• 数据指针访问 `data_ptr()` 禁止用于符号类型
