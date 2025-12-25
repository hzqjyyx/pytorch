这个文件定义了一系列条件编译宏，用于在编译时检测 CUDA 和 cuSparse 库的版本特性，以实现版本兼容性。

主要内容：

- **cuSparse Generic API 支持**：检测 CUDA 10.1+ 和 cuSparse 10.3+ (或 Windows 11.0+)，定义 `AT_USE_CUSPARSE_GENERIC_API()` 宏

- **cuSparse 描述符常量性**：针对 CUDA 12.0 的 API 变更，区分 const 和 non-const 描述符指针，定义 `AT_USE_CUSPARSE_CONST_DESCRIPTORS()` 和 `AT_USE_CUSPARSE_NON_CONST_DESCRIPTORS()` 宏

- **cuSparse 稀疏线性求解函数**：
  - `AT_USE_CUSPARSE_GENERIC_SPSV()` - CUDA 11.5+ 的向量稀疏三角求解
  - `AT_USE_CUSPARSE_GENERIC_SPSM()` - CUDA 11.6+ 的矩阵稀疏三角求解
  - `AT_USE_CUSPARSE_GENERIC_SDDMM()` - CUDA 11.4+ 的稠密-稀疏-稠密矩阵乘法

- **用途**：允许 ATen 库在编译时根据可用的 CUDA 版本特性选择最优的稀疏矩阵运算实现，提高性能和功能完整性
