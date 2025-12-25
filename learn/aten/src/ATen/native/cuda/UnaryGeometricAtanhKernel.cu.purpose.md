这个文件实现了 PyTorch 中 `atanh`（反双曲正切）函数的 CUDA 内核。

**主要功能：**

- **处理复数类型**：如果输入是复数（包括 `kComplexHalf`），使用 JIT 编译器生成 GPU 内核，或直接调用 `std::atanh`
- **处理浮点类型**：支持 `Half`、`BFloat16` 和标准浮点类型，通过 `gpu_kernel` 执行并行计算
- **OpMath 类型转换**：复数类型的计算中，先将输入转换为 OpMath 类型（更高精度），再执行 `atanh` 运算
- **TensorIterator 接口**：通过 `TensorIterator` 统一处理多维张量和广播操作
- **核心运算**：对每个元素计算 `atanh(a) = 0.5 * ln((1+a)/(1-a))`，即反双曲正切函数

**关键点：**

- 使用 `AT_DISPATCH_*` 宏进行类型分发
- `GPU_LAMBDA` 定义了在 GPU 上执行的计算逻辑
- 通过 `REGISTER_DISPATCH` 注册到全局分发表，供上层 API 调用
