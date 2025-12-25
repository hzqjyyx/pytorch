该文件定义了 PyTorch 损失函数的 reduction 行为常数：

- **命名空间**：`at::Reduction`
- **枚举类型**：`Reduction` 包含三种 reduction 模式
  - `None`：不进行 reduction，保持原始损失值
  - `Mean`：计算加权或未加权的平均损失
  - `Sum`：对所有损失进行求和
  - `END`：枚举结束标记
- **同步要求**：需要与 `torch/nn/_reduction.py` 中的对应类保持同步
- **设计限制**：采用普通 enum 而非 scoped enum，原因是 JIT 编译器不支持 scoped enum
