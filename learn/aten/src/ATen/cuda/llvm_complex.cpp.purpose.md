这个文件提供了用于 CUDA JIT 编译的 C++ 复数类型实现，核心是将 LLVM libc++ 的 `std::complex` 实现以字符串形式嵌入，供运行时代码生成使用。

## 主要组成部分

**1. 基础复数模板类 (`complex_body` 字符串)**
- 通用模板 `complex<_Tp>`: 支持任意类型的复数，包含实部/虚部访问、赋值、复合赋值运算符
- 特化版本 `complex<float>` 和 `complex<double>`: 为浮点类型提供优化实现
- 相互转换构造函数，支持不同精度间转换

**2. 复数算术运算符**
- 加减乘除四则运算，支持复数-复数、复数-标量的各种组合
- 乘法实现特殊处理 NaN/Inf 边界情况（lines 286-343）
  - 当结果为 NaN 时检测操作数是否为无穷大
  - 使用 `copysign` 规范化无穷大操作数为 ±1
  - 重新计算得到正确的无穷大结果
- 除法采用数值稳定算法（lines 365-407）
  - 使用 `logb`/`scalbn` 进行尺度归一化避免上溢/下溢
  - 特殊处理除零、无穷大分子/分母等边界情况

**3. 复数数学函数 (`complex_math` 字符串)**
基本函数：
- `abs`, `arg`, `norm`, `conj`, `proj`: 幅值、辐角、范数、共轭、投影

代数函数：
- `sqrt`, `exp`, `log`, `log10`, `log2`, `pow`: 使用极坐标形式计算

三角函数：
- `sin`, `cos`, `tan`: 通过双曲函数转换实现
  ```cpp
  sin(z) = sinh(complex(-z.imag(), z.real())).imag_rotated()
  ```

双曲函数：
- `sinh`, `cosh`, `tanh`: 使用实部/虚部分解公式
  ```cpp
  sinh(a+bi) = sinh(a)cos(b) + i·cosh(a)sin(b)
  ```

反函数：
- `asin`, `acos`, `atan`: 通过反双曲函数转换
- `asinh`, `acosh`, `atanh`: 使用对数形式定义
  ```cpp
  atanh(z) = log((1+z)/(1-z)) / 2
  ```

所有函数都包含完整的 NaN/Inf 传播逻辑，遵循 IEEE 754 标准。

**4. Half 精度支持 (`complex_half_body` 字符串)**
```cpp
template<> struct complex<at::Half> {
  at::Half real_, imag_;
  // 隐式转换到 complex<float> 进行计算
  operator std::complex<float>() const;
}
```
通过提升到 `float` 精度进行运算，自动处理精度转换。

**5. 工具设施**
- `__libcpp_complex_overload_traits`: 类型萃取，整数映射到 `double`，浮点保持原类型
- 字面量后缀 `operator""i`/`operator""if`: 支持 `1.0i` 语法
- 比较运算符、逻辑运算符（`==`, `!=`, `&&`, `||`）

## 使用场景
这些字符串被 `get_complex_body_string()` 等函数返回，在 CUDA JIT 编译时注入到生成的内核代码中，使得 CUDA kernel 可以使用与标准库兼容的复数类型。

---

**简略列出的其他内容：**
• 文件头部声明来源于 LLVM 项目（Apache 2.0 许可证）  
• 命名空间为 `at::cuda`  
• 不涉及 ROCm 平台特定代码  
• 不包含 backward（反向传播）相关实现
