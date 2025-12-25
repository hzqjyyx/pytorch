这个文件定义了量化相关操作的函数指针类型和分发声明。

**主要功能：**

- **fake_quant_tensor_cachemask_fn**: 对张量进行仿真量化，缓存量化掩码，接收标量缩放因子和零点参数
- **fake_quant_tensor_cachemask_tensor_qparams_fn**: 对张量进行仿真量化，量化参数（缩放因子、零点）本身为张量形式
- **fake_quant_learnable_grad_tensor_fn**: 可学习的仿真量化梯度计算，支持梯度因子调整
- **fake_quant_per_channel_fn**: 逐通道仿真量化操作
- **fake_quant_per_channel_cachemask_fn**: 逐通道仿真量化并缓存掩码
- **fake_quant_learnable_per_channel_fn**: 可学习的逐通道仿真量化梯度计算
- **DECLARE_DISPATCH**: 声明这些函数的后端分发机制，支持不同硬件实现（CPU、CUDA等）
