**MaxPooling.cpp 主要功能：**

- `max_pool1d_impl()` 函数：执行1D最大池化的核心实现
  - 计算输出形状 (OW)
  - 创建空输出张量
  - 构建 `PoolingParams1D` 参数结构体
  - 通过 `max_pool1d_stub` 调度到设备特定的实现 (CPU/CUDA)
  - 处理2D输入时的squeeze操作
  - 传播张量名称信息

- `max_pool1d()` 函数：公共API接口
  - 验证输入张量维度 (2D或3D)
  - 处理量化张量的特殊情况
  - 调用参数检查函数
  - 根据条件选择执行路径：
    - 需要梯度、forward gradient、非CPU设备或子类张量 → 调用 `max_pool1d_with_indices()` (支持反向传播)
    - 其他情况 → 调用 `max_pool1d_impl()` (简化路径)

**MaxPooling.h 主要功能：**

- `check_max_pool1d()` 函数：参数验证
  - 检查输入维度 (2D或3D)
  - 验证 kernel_size、stride、padding、dilation 的大小
  - 验证各参数值的有效性 (大于0、非负等)
  - 检查 padding 不超过 kernel_size 的一半
  - 计算并验证输出大小 > 0

- `PoolingParams1D` 结构体：1D池化参数容器
  - 存储批次数、通道数、输入/输出宽度、卷积核宽度、步长、填充、膨胀
  - `index()` 方法：计算给定卷积核和输出索引对应的输入元素索引
  - `valid_output_start()` 方法：计算该卷积核位置有效输出的起始索引
  - `valid_output_end()` 方法：计算该卷积核位置有效输出的结束索引

- `max_pool1d_stub`：调度机制
  - 声明函数指针类型 `pooling_fn`
  - 为不同设备后端提供实现注册点
