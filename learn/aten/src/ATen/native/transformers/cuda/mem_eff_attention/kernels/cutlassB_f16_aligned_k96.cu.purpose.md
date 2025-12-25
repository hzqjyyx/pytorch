这是一个自动生成的 CUDA 内核文件，专门为高效内存注意力机制的反向传播计算优化。

**主要功能：**

- **文件用途**：实现 FMHA（Flash Memory-Efficient Attention）反向传播内核，通过 `generate_kernels.py` 自动生成
- **目标架构**：SM80（NVIDIA A100 等新一代 GPU），支持 SM80-SM100 计算能力范围
- **内核配置**：
  - 数据类型：float16（半精度浮点数）
  - 线程块大小：128×64
  - 头维度（K）：96
  - 对齐模式：aligned（内存对齐优化）
- **核心操作**：调用 `AttentionBackwardKernel` 执行反向传播梯度计算
- **安全检查**：
  - 编译期检查 CUDA 架构版本
  - 运行期验证 GPU 计算能力，不匹配时输出错误信息
  - 动态跳过不需要处理的线程块（`advance_to_block()`）
- **优化特点**：通过 `__launch_bounds__` 指定线程束调度参数以最大化 GPU 利用率
