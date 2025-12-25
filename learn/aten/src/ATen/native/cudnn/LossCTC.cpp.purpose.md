这个文件实现了 CTC Loss（Connectionist Temporal Classification Loss）的 CUDA 加速计算，使用 cuDNN 库。

## 核心功能分解

**条件编译结构：**
- 当 cuDNN 未启用时，函数返回 false 或抛出错误
- 当 cuDNN 启用时，提供完整实现

**两套 API 设计：**

1. **IntArrayRef 版本** - 接收数组引用作为长度参数
   - `_use_cudnn_ctc_loss()` - 检查是否可使用 cuDNN
   - `_cudnn_ctc_loss()` - 执行 CTC Loss 计算（cuDNN 7.6）

2. **Tensor 版本** - 接收 Tensor 作为长度参数（更灵活）
   - `_use_cudnn_ctc_loss_tensor()` - 检查是否可使用 cuDNN
   - `_cudnn_ctc_loss_tensor()` - 执行 CTC Loss 计算（cuDNN 8.x+）

**可用性检查条件：**
- BLANK 标签必须为 0
- log_probs：Float 类型、3D、GPU 上
- targets：Int 类型、1D、连续内存、CPU 上
- 目标长度 < 256 且 ≤ 输入长度

**计算流程：**
- 创建 CTC Loss 描述符（设置归一化、NaN 处理）
- 查询 cuDNN 所需的工作空间大小
- 分配梯度和成本张量
- 调用 cuDNN 核函数计算 CTC Loss 和梯度
- 返回 (costs, grad) 元组

**关键特性：**
- 支持确定性/非确定性算法选择
- 支持 CUDA Graph 捕获环境
- Tensor 版本支持异步断言检查

---

**总结：**

- 实现 CTC Loss 的 cuDNN 加速前向/反向计算
- 提供两套 API（数组和 Tensor 参数版本）
- 包含广泛的输入验证和可用性检查
- 支持不同 cuDNN 版本（7.6 vs 8.x+）的兼容性
- 返回成本和梯度用于模型训练
