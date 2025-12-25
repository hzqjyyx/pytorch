这个文件实现了 `Float8_e5m2fnuz` 类型的内联函数定义，提供 8 位浮点数运算支持。

## 核心功能

### 1. 类型转换
- **构造函数** (line 17-18): 从 `float` 转换为 `Float8_e5m2fnuz`，调用 `fp8e5m2fnuz_from_fp32_value`
- **隐式转换** (line 22-24): `Float8_e5m2fnuz` 转换回 `float`，调用 `fp8_fnuz_to_fp32_value<5, 2>`

### 2. 特殊值检测
- **isnan()** (line 28-30): 检查是否为 NaN，判断位模式是否为 `0b10000000`
- **isinf()** (line 32-34): 始终返回 `false`，因为 fnuz 格式不支持无穷大

### 3. 运算符重载

所有算术运算都通过先转换为 `float` 进行计算，然后转换回 `Float8_e5m2fnuz`：

**同类型运算** (line 38-89):
- 二元运算: `+`, `-`, `*`, `/`
- 一元运算: `-`
- 复合赋值: `+=`, `-=`, `*=`, `/=`

**与其他类型混合运算**:
- **float 混合** (line 93-132): 返回 `float`，支持双向运算和复合赋值
- **double 混合** (line 136-162): 返回 `double`，支持双向运算
- **int 混合** (line 166-190): 返回 `Float8_e5m2fnuz`，支持双向运算
- **int64_t 混合** (line 194-218): 返回 `Float8_e5m2fnuz`，支持双向运算

### 4. std::numeric_limits 特化 (line 227-281)

定义了 `Float8_e5m2fnuz` 的数值极限特性：

**格式参数**:
- `digits = 3`: 尾数位数（含隐藏位）
- `min_exponent = -14`, `max_exponent = 16`: 指数范围
- `has_infinity = false`, `has_quiet_NaN = true`: 不支持无穷，支持 NaN

**特殊值**（通过位模式构造）:
- `min()`: `0x04` - 最小正规化数
- `max()`: `0x7F` - 最大值  
- `lowest()`: `0xFF` - 最小负值
- `epsilon()`: `0x34` - 机器精silon
- `quiet_NaN()` / `infinity()`: 都是 `0x80`（负零位模式，存在歧义，见 line 273-274 注释）

### 5. UBSan 注解
除法运算使用 `__ubsan_ignore_float_divide_by_zero__` 属性，避免除零时的 UBSan 检查

---

**ROCm/Backward 相关**: 无
