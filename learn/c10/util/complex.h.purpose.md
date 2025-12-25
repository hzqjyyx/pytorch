# c10::complex - PyTorch 跨设备复数实现

这个文件实现了 `c10::complex<T>` 模板类，目标是在 PyTorch 支持的所有设备（CPU、CUDA、ROCm）上提供统一的复数运算。

## 核心设计

### 数据布局
```cpp
template <typename T>
struct alignas(sizeof(T) * 2) complex {
  T real_ = T(0);
  T imag_ = T(0);
};
```
- 使用两个标量成员 `real_` 和 `imag_` 存储实部和虚部
- 对齐到 `sizeof(T) * 2`，确保内存布局符合硬件要求

### 类型转换系统

**三种复数类型的互操作**：
- `std::complex<T>` - C++ 标准库，仅 CPU
- `thrust::complex<T>` - NVIDIA Thrust 库，CUDA/HIP 设备
- `c10::complex<T>` - PyTorch 自有实现，跨平台

支持的转换：
```cpp
// 从 std::complex 构造（显式）
explicit constexpr complex(const std::complex<U>& other);

// 从 thrust::complex 构造（显式，CUDA/HIP）
explicit C10_HOST_DEVICE complex(const thrust::complex<U>& other);

// 转换回 std::complex
explicit constexpr operator std::complex<U>() const;

// 转换回 thrust::complex
C10_HOST_DEVICE explicit operator thrust::complex<U>() const;
```

**精度转换**：
- `complex<float>` ↔ `complex<double>` 之间可以相互转换
- double→float 是显式转换（防止精度损失）
- float→double 是隐式转换（安全提升）

## 运算符实现

### 复合赋值运算符
```cpp
complex& operator+=(const complex& rhs);  // 加法
complex& operator-=(const complex& rhs);  // 减法
complex& operator*=(const complex& rhs);  // 乘法
complex& operator/=(const complex& rhs);  // 除法
```

**除法实现细节**（c10/util/complex.h:246-282）：
- 采用 NumPy 的数值稳定算法
- 根据 `|c|` vs `|d|` 选择不同计算路径，避免溢出
- 特殊处理除零情况（产生 inf/nan）

### 整数运算优化
```cpp
template <typename fT, typename iT>
constexpr c10::complex<fT> operator+(const c10::complex<fT>& a, const iT& b);
```
- 允许复数与整数直接运算，无需显式类型转换
- 提升代码简洁性和潜在性能

### 布尔转换
```cpp
explicit constexpr operator bool() const {
  return real() || imag();
}
```
- 与 NumPy 行为一致：实部或虚部非零即为 true

## 标准库函数扩展

在 `std` 命名空间中为 `c10::complex` 提供标准函数：

```cpp
std::real(z)    // 获取实部
std::imag(z)    // 获取虚部
std::abs(z)     // 模长：√(real² + imag²)
std::arg(z)     // 幅角：atan2(imag, real)
std::norm(z)    // 范数：real² + imag²
std::conj(z)    // 共轭：real - i*imag
```

### 设备兼容实现
```cpp
template <typename T>
C10_HOST_DEVICE T abs(const c10::complex<T>& z) {
#if defined(__CUDACC__) || defined(__HIPCC__)
  return thrust::abs(static_cast<thrust::complex<T>>(z));
#else
  return std::abs(static_cast<std::complex<T>>(z));
#endif
}
```
- CPU 路径使用 `std::abs`
- GPU 路径使用 `thrust::abs`

## c10 命名空间扩展

### polar 函数
```cpp
template <typename T>
C10_HOST_DEVICE complex<T> polar(const T& r, const T& theta = T());
```
- 从极坐标 (r, θ) 构造复数
- 返回 `c10::complex`（而非 `std::complex`）

### 自定义字面量
```cpp
using namespace c10::complex_literals;
auto z1 = 3.14_if;  // complex<float>(0.0f, 3.14f)
auto z2 = 2.71_id;  // complex<double>(0.0, 2.71)
```

## Half 精度特化

`c10::complex<Half>` 的特殊实现（c10/util/complex.h:611-657）：
```cpp
template <>
struct alignas(4) complex<Half> {
  Half real_;
  Half imag_;
  
  // 构造函数不是 constexpr（因为 Half 限制）
  C10_HOST_DEVICE explicit inline complex(const Half& real, const Half& imag);
  
  // 提供与 complex<float> 的转换
  C10_HOST_DEVICE inline complex(const c10::complex<float>& value);
  inline C10_HOST_DEVICE operator c10::complex<float>() const;
};
```

**运算实现策略**：
- 内部转换为 `float` 进行计算
- 结果转换回 `Half`
- 例如加法：`real_ = float(real_) + float(other.real_)`

## I/O 流操作

```cpp
std::ostream& operator<<(std::ostream& os, const complex<T>& x);
std::istream& operator>>(std::istream& is, complex<T>& x);
```
- 通过转换为 `std::complex<T>` 实现
- 输出格式与标准库一致：`(real,imag)`

---

**其他特性**：
- ROCm 平台的 `atan2` bug 工作区（`ROCm_Bug` 宏）
- 向后兼容：保留 C++20 前已移除的 `operator!=` 重载
- 对齐控制：通过 `alignas` 确保内存布局正确性
- FORCE_INLINE_APPLE：macOS 平台的除法运算强制内联优化
- UBSan 属性：`__ubsan_ignore_float_divide_by_zero__` 允许合法的浮点除零行为
