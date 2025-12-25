这个文件定义了 PyTorch Flash Attention 在 HIP/ROCm 平台上的 C++ API 接口。

**核心功能：**

- **前向传播函数** (`mem_eff_forward_ck`): 实现内存高效的 Flash Attention 前向计算
  - 输入: Q、K、V 张量，dropout 概率，因果掩码标志，注意力偏置等
  - 输出: 注意力输出、归一化指数和(LSE)、dropout 随机值等中间结果
  - 支持可变长度序列(通过 `cu_seqlens_q/k`)

- **后向传播函数** (`mem_eff_backward_ck`): 计算梯度
  - 输入: 输出梯度、前向传播的所有张量和中间结果
  - 输出: Q、K、V 的梯度以及注意力偏置梯度

**关键特性：**

- 条件编译: 仅在 `USE_CK_FLASH_ATTENTION` 定义时包含
- 支持 dropout 和随机种子管理(Philox RNG)
- 支持因果 Attention(用于自回归模型)
- 支持自定义 attention 缩放因子
- 支持可变批处理大小和序列长度

**架构角色：**

- 作为 CK(Composable Kernel)库的 PyTorch 适配层
- 桥接 PyTorch 张量类型与底层 HIP 计算核心
