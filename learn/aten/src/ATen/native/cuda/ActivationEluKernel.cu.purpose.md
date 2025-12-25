**ELU 激活函数 CUDA 核心实现**

该文件实现了 ELU (Exponential Linear Unit) 激活函数的 CUDA GPU 版本。

**核心组件：**

- **elu_kernel** (第22-45行)
  - 前向传播实现
  - 对正数：返回 `input * scale`
  - 对负数：返回 `(exp(input * input_scale) - 1) * alpha * scale`
  - 使用 `gpu_kernel` 并行处理张量中的每个元素

- **elu_backward_kernel** (第47-80行)
  - 反向传播（梯度计算）
  - 支持两种模式（`is_result` 标志）：
    - 若输入是 ELU 输出结果：直接用结果计算梯度
    - 若输入是原始输入值：需要指数计算梯度
  - 正数梯度：`grad_input * scale`
  - 负数梯度：涉及 `exp` 或指数项

- **关键特性：**
  - 支持 Float32、Float16、BFloat16 等多种数据类型
  - 使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 宏进行类型分发
  - 采用 `opmath_t` 提高数值精度（高精度中间计算）
  - 通过 `REGISTER_DISPATCH` 注册为 PyTorch 可调用的操作

**功能列表：**
- ELU 前向激活函数的 GPU 实现
- ELU 反向梯度计算
- 支持参数化的 alpha、scale、input_scale
- 多种浮点数据类型支持
- 高精度计算确保数值稳定性
