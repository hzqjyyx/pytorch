这两个文件实现了 PyTorch 中的融合 Adam 优化器的 CPU 分发层。

**FusedAdam.h** 定义了核心接口：
- `ADAM_MODE` 枚举区分两种优化器模式（ORIGINAL Adam 和 AdamW）
- `fused_adam_fn` 函数指针类型定义了优化器的通用签名
- `DECLARE_DISPATCH` 声明了 `fused_adam_stub`，用于分发到不同后端实现

**FusedAdam.cpp** 提供了 CPU 实现的包装层：
- 两个 `_fused_adam_kernel_cpu_` 重载：一个接受 double 类型的 lr，一个接受 Tensor 类型的 lr（内部调用第一个）
- 两个 `_fused_adamw_kernel_cpu_` 重载：结构相同，但使用 `ADAM_MODE::ADAMW` 模式
- 共同逻辑：
  - 处理可选的梯度缩放（grad_scale）和溢出检测（found_inf）
  - 验证所有输入张量列表大小一致
  - 逐个处理参数列表，调用 `fused_adam_stub` 进行实际优化计算
  - 在检测到溢出（found_inf=1.0）时提前返回，跳过优化步骤

**主要功能点：**
- 为 Adam 和 AdamW 优化器提供 CPU 后端的分发接口
- 支持梯度缩放和数值稳定性检查（溢出检测）
- 使用分发机制将 CPU 计算委托给专门实现
- 处理学习率的灵活性（标量或张量形式）
- 支持 AMSGrad 变体选项
