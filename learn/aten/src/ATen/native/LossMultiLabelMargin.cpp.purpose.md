## LossMultiLabelMargin.cpp 主要功能分析

这个文件实现了多标签边界损失（Multi-Label Margin Loss）的CPU计算逻辑。

### 核心算法

**multilabel_margin_loss_forward_inner_sum_cpu**（26-58行）：
- 接收输入scores、目标标签索引、标记数组
- 第一遍循环：标记哪些索引是目标标签（is_target_data[target_idx] = 1）
- 第二遍循环：对每个目标标签，遍历所有非目标位置，计算margin loss: `z = 1 - input_target + input_data[d]`
- 当z > 0时累加到损失中

**multilabel_margin_loss_forward_out_frame**（61-109行）：
- 处理单个或批量数据的前向传播
- 支持两种输出模式：
  - 若reduction != None或输出是标量：累加所有样本损失，按dim和nframe归一化
  - 否则：为每个样本单独计算损失，存储为1D张量

**multilabel_margin_loss_backward_out_frame**（156-223行）：
- 计算反向传播梯度
- 遍历正向传播的相同逻辑，对激活的margin项（z > 0）更新梯度
- 目标标签位置梯度减g，非目标位置梯度加g
- 最后乘以grad_output进行链式法则传播

### 关键设计

- 使用**累积标量类型**（acc_type）处理数值精度问题
- 支持**内存高效**的行指针遍历（input_data += dim）
- 使用**accessor接口**处理张量访问
- 通过AT_DISPATCH_FLOATING_TYPES宏实现浮点类型的自动分发

### 输出汇总

- **Forward Pass**: 计算multi-label margin loss，支持Mean/Sum/None三种reduction模式
- **Backward Pass**: 对应的梯度计算
- **Public API**: 
  - `multilabel_margin_loss_forward_out_cpu` - 输出模式前向
  - `multilabel_margin_loss_forward_cpu` - 普通模式前向
  - `multilabel_margin_loss_backward_cpu_out` - 输出模式反向
  - `multilabel_margin_loss_backward_cpu` - 普通模式反向
  - `multilabel_margin_loss_out` / `multilabel_margin_loss` - 统一接口
