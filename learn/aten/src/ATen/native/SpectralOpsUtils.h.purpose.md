这个文件定义了 PyTorch 的 FFT（快速傅里叶变换）操作的工具函数和类型。

**主要功能：**

- **FFT 归一化模式** (`fft_norm_mode` enum)：定义三种归一化选项
  - `none`：不归一化
  - `by_root_n`：除以信号大小的平方根
  - `by_n`：除以信号大小

- **共轭对称性处理**：实数到复数的 FFT 满足共轭对称性，库通常只返回约一半的值来避免冗余

- **大小推断函数**：
  - `infer_ft_real_to_complex_onesided_size()`：从实数信号大小推断单边复数频域大小
  - `infer_ft_complex_to_real_onesided_size()`：从单边复数大小反推实数信号大小（支持可选的预期大小验证）

- **共轭对称性填充**：
  - `fft_fill_with_conjugate_symmetry_stub`：函数指针类型和分发机制
  - `_fft_fill_with_conjugate_symmetry_()`：利用 Hermitian 对称性填充 FFT 的另一半值

- **跨平台兼容性**：设计考虑了 cuFFT 和 MKL 等不同 FFT 库的约定
