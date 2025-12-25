这个文件是 BFloat16（Brain Floating Point 16-bit）类型的内联实现文件，提供了完整的运算符重载和类型转换支持。

## 核心功能

**1. 构造和类型转换**
- 从 float 构造 BFloat16，在 CUDA Ampere（sm_80+）上使用硬件指令 `__float2bfloat16`，在 SYCL 上使用 oneapi 扩展，其他平台使用软件实现的 round-to-nearest-even（RNE）算法
- 隐式转换回 float，CUDA 上用 `__bfloat162float` 硬件指令，SYCL 用 oneapi 类型转换，其他平台用 `f32_from_bits` 从 16-bit 表示重构
- 与 CUDA `__nv_bfloat16` 和 SYCL `sycl::ext::oneapi::bfloat16` 原生类型的双向转换

**2. 算术运算符（BFloat16 ↔ BFloat16）**
所有运算都先转换为 float 执行，再转回 BFloat16：
- 二元运算：`+`, `-`, `*`, `/`（除法标记 `__ubsan_ignore_float_divide_by_zero__`）
- 一元运算：`-`（取负）
- 复合赋值：`+=`, `-=`, `*=`, `/=`
- 位运算：`|`, `^`, `&`（直接操作底层 16-bit 表示 `x`）

**3. 混合类型算术**
- **BFloat16 ↔ float**: 返回 float，避免精度损失
- **BFloat16 ↔ double**: 返回 double，BFloat16 先转 double 再运算
- **BFloat16 ↔ int/int64_t**: 返回 BFloat16，整数先转 BFloat16 再运算
- 所有混合类型都实现了双向运算（a op b 和 b op a）

**4. 比较运算符**
- `operator<` 和 `operator>`：转换为 float 后比较，用于支持 `std::max`/`std::min`
- 参数是非 const 引用（这是为了匹配标准库期望的签名）

**5. CUDA 特殊支持**
- `__ldg` 函数：在 sm_80+ 上使用 load-global 指令优化从全局内存读取

**6. std::numeric_limits 特化**
完整定义了 BFloat16 的数值极限：
- **精度参数**: 8 位尾数（digits=8），2 位十进制精度（digits10=2）
- **指数范围**: [-125, 128]（对应十进制 [-37, 38]）
- **特殊值**: 通过 `from_bits()` 构造函数直接指定位模式
  - `min()`: 0x0080（最小正规数）
  - `max()`: 0x7F7F（最大有限数）
  - `epsilon()`: 0x3C00（机器精度）
  - `infinity()`: 0x7F80
  - `quiet_NaN()`: 0x7FC0
  - `denorm_min()`: 0x0001（最小非规格化数）

## 设计特点

- **平台适配**: 通过预处理器宏在 CUDA、SYCL、通用 CPU 之间切换实现
- **性能优化**: 在支持硬件的平台上使用原生指令，否则回退到软件实现
- **标准库兼容**: 提供 `numeric_limits` 特化和比较运算符，使其能用于 STL 容器和算法
- **精度策略**: BFloat16 间运算提升到 float 执行，与 float/double 混合运算直接提升到高精度类型

---

**忽略的内容：**
- ROCm 相关：`USE_ROCM` 宏分支（lines 27-28, 42-59, 73-80）
- Backward compatibility：无明显的向后兼容代码
