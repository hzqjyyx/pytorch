这个文件定义了用于混合精度 GEMM（通用矩阵乘法）操作的架构特征模板。

**主要功能：**

- **MixedGemmArchTraits 模板结构**：为不同 GPU 架构和数据类型组合提供特定的计算参数和配置

- **支持的架构**：
  - 通用浮点 (float)
  - Volta (Sm70)：使用 HMMA tensorcore，fp16 计算
  - Turing (Sm75)：使用 TensorOp，fp16 计算
  - Ampere (Sm80)：使用 TensorOp，fp16 计算

- **关键配置参数**：
  - `Stages`：pipeline 阶段数
  - `OperatorClass`：操作类型（SIMT 或 TensorOp）
  - `AccType`：累积类型（accumulator）
  - `LayoutB`：矩阵 B 的内存布局
  - `ElementsPerAccessA/B/C`：单次访问的元素数
  - `ThreadblockK`：线程块处理的 K 维度
  - `InstructionShape`：GEMM 指令的计算形状

- **设计模式**：使用模板特化（template specialization）为不同硬件提供优化配置，自动选择最优的矩阵布局和计算参数
