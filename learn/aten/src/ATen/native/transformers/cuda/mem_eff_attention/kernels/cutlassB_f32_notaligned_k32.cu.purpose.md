这是一个 PyTorch CUDA 内存高效注意力机制（Memory-Efficient Attention）的**反向传播内核**生成文件。

**主要功能：**

- **自动生成文件**：由 `generate_kernels.py` 脚本生成，包含预编译的 CUDA 内核代码
- **多架构支持**：为不同 GPU 计算能力（SM 50/70/75）分别实现优化的反向传播内核
- **内核函数**：
  - `fmha_cutlassB_f32_notaligned_64x64_k32_sm50`：SM 50-70 架构
  - `fmha_cutlassB_f32_notaligned_64x64_k32_sm70`：SM 70-75 架构
  - `fmha_cutlassB_f32_notaligned_64x64_k32_sm75`：SM 75-80 架构
- **参数配置**：使用 CUTLASS 库的 `AttentionBackwardKernel` 模板，float32 精度，无对齐优化，tile 大小 64×64，head dimension k=32
- **运行时检查**：使用 `__CUDA_ARCH__` 宏在编译时验证代码是否在正确的 SM 版本上运行，不匹配则打印错误信息
- **性能优化**：通过 `__launch_bounds__` 指定线程块大小和最小 SM 占用率以获得最优性能
