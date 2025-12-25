# CUDADataType.h 主要功能

这个文件提供了 PyTorch CUDA 数据类型转换的核心功能，包含两个主要部分：

## 1. 模板函数 `getCudaDataType<T>()`
通过模板特化为 C++ 标量类型映射到 CUDA 数据类型：
- 浮点类型：Half → CUDA_R_16F、float → CUDA_R_32F、double → CUDA_R_64F
- 复数类型：complex<Half/float/double> → CUDA_C_16F/32F/64F
- 整数类型：uint8_t/int8_t/int16_t/int/int64_t → 对应的 CUDA_R_*I 类型
- BFloat16 → CUDA_R_16BF

默认模板会触发编译时断言，防止未支持类型的隐式转换。

## 2. 函数 `ScalarTypeToCudaDataType()`
运行时将 PyTorch 的 ScalarType 枚举值转换为 CUDA 数据类型：
- 处理所有 PyTorch 标量类型（Byte、Char、Int、Float、Double 等）
- 提供条件编译支持 Float8 类型（CUDA 11.8+ 或 ROCm 6.3+）
- 非法类型输入触发运行时断言

---

**核心特点：**
- 两层转换机制：编译期类型安全（模板）+ 运行期灵活性（枚举）
- 保证类型安全，防止未支持的隐式转换
- 为 cuBLAS、cuSPARSE 等 CUDA 库的 API 调用提供类型适配
