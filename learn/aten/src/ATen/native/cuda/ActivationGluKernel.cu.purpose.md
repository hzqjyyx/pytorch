## 主要功能

这个文件实现了 PyTorch 中 GLU（Gated Linear Unit）激活函数的 CUDA 核心操作。

### 核心组件

**GLU Forward (glu_kernel, 24-36行)**
- 对两个输入张量进行元素级操作
- 计算公式：`output = a * sigmoid(b)`
- 支持 float32、float16、bfloat16 数据类型
- 使用 `gpu_kernel` 在 GPU 上并行执行

**GLU JVP (glu_jvp_kernel, 41-60行)**
- 前向模式自动微分（Jacobian-Vector Product）
- 计算梯度传播：`d_output = da * sigmoid(b) + a * sigmoid'(b) * db`
- 用于支持高阶导数计算

**GLU Backward (glu_backward_kernel, 76-107行)**
- 反向传播实现
- 计算对两个输入的梯度：
  - `grad_a = sigmoid(b) * grad_output`
  - `grad_b = (1 - sigmoid(b)) * sigmoid(b) * grad_output * a`
- 使用 `byte_offset` 优化内存访问
- 自定义 CUDA 核函数实现高效计算

**启动器函数 (launch_glu_backward_kernel, 109-136行)**
- 配置 CUDA 网格和线程块参数
- 线程块大小：256
- 处理动态内存偏移计算

### 关键特性

- **类型泛化**：通过 `AT_DISPATCH_FLOATING_TYPES_AND2` 支持多种浮点类型
- **操作数学类型**：使用 `opmath_type` 提高计算精度
- **高效内存访问**：字节偏移减少 64 位索引开销
- **自动微分支持**：完整的前向和反向自动微分

### 核心清单

- GLU 激活函数前向计算（sigmoid 门控线性单元）
- 前向模式自动微分支持
- 反向传播梯度计算
- CUDA 并行优化实现
- 多数据类型支持（float32/16, bfloat16）
