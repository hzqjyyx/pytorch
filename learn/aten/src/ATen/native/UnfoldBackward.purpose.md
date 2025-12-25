**UnfoldBackward.cpp** - 主要实现函数

`unfold_backward()` 函数计算 unfold 操作的反向梯度：
- 输入：梯度张量、原始输入尺寸、展开维度、窗口大小、步长
- 创建零初始化的梯度输入张量
- 当 `step >= size` 时，直接用 unfold 和 copy 操作快速路径
- 否则调用设备特定的 stub 函数处理复杂情况

**UnfoldBackward.h** - 声明和工具函数

`unfold_backward_fn` 函数指针类型定义 stub 接口

`_make_unfold_backward_iter_over_grad_out()` 辅助函数构造 TensorIterator：
- 处理维度包装和空张量
- 计算迭代维度大小：`min(grad_out_dim_size, (grad_in_dim_size - 1) * step + size)`
- 为 TensorIterator 准备三个输入：
  - grad_out（重新步长化）
  - grad_in（降维后重新步长化，dim 步长设为 0）
  - idx_dim（坐标索引张量，按 dim 广播）
- 配置 TensorIterator 跳过内存重叠检查、保留输出形状

---

**关键点总结：**

- 反向传播梯度累积到输入梯度张量
- 快速路径：step ≥ size 时可直接复制
- 复杂路径：使用 TensorIterator 遍历，设备层实现细节由 stub 负责
- TensorIterator 巧妙利用步长和广播处理不规则的梯度分散操作
