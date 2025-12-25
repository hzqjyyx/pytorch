这个文件是 PyTorch ATen 库的配置头文件模板。它使用 CMake 的模板替换机制（`@VARIABLE@` 语法）来定义编译时的条件特性宏。

**主要功能：**

- **MKL 相关**：定义 MKL（Intel Math Kernel Library）是否启用及其工作模式（`AT_MKL_ENABLED`、`AT_MKL_SEQUENTIAL`）
- **MKLDNN 支持**：控制深度学习优化库的启用（`AT_MKLDNN_ENABLED`、`AT_MKLDNN_ACL_ENABLED`）
- **FFT 库**：启用 PocketFFT 快速傅里叶变换库（`AT_POCKETFFT_ENABLED`）
- **神经网络加速**：NNPACK 库支持（`AT_NNPACK_ENABLED`）
- **BLAS/LAPACK**：线性代数库的启用和配置（`AT_BUILD_WITH_BLAS`、`AT_BUILD_WITH_LAPACK`、`AT_BLAS_F2C`、`AT_BLAS_USE_CBLAS_DOT`）
- **并行计算**：OpenMP 和原生并行选项（`AT_PARALLEL_OPENMP`、`AT_PARALLEL_NATIVE`）
- **CUDA 配置**：静态链接 CUDA 的控制（`CAFFE2_STATIC_LINK_CUDA`）
- **Kleidi AI**：ARM 优化库支持（`AT_KLEIDIAI_ENABLED`）

**设计原则**：
- 使用 `#define` 而非 `#ifdef` 以确保所有宏都被显式定义，避免意外的未定义检查错误
- 采用函数宏形式（如 `AT_MKL_ENABLED()`）以便在条件编译中使用
