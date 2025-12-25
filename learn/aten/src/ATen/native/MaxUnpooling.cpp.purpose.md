**主要功能：**

- **max_unpooling2d_forward_out_cpu**: CPU 上 2D 最大反池化的核心实现
  - 验证输入张量维度（3D 或 4D）、索引类型（int64）、输出尺寸参数
  - 检查输入和索引形状一致性、非零维度
  - 调用 `max_unpool2d_kernel` 执行实际反池化操作
  - 支持内存格式优化（NHWC/NCHW）

- **max_unpooling2d_forward_cpu**: 2D 反池化的包装函数
  - 创建空输出张量，委托给 out 版本处理

- **max_unpooling3d_shape_check**: 3D 反池化的形状检查函数
  - 验证 4D/5D 输入、stride/padding/output_size 参数合法性
  - 验证索引类型和尺寸一致性
  - 可选验证梯度输出尺寸

- **max_unpooling3d_forward_out_cpu**: CPU 上 3D 最大反池化实现
  - 类似 2D 版本的流程（验证→调整大小→零初始化→内核调用）
  - 调用 `max_unpool3d_kernel` 执行反池化

- **max_unpooling3d_forward_cpu**: 3D 反池化的包装函数
  - 创建空输出张量，委托给 out 版本处理

- **调度宏**: 使用 `DEFINE_DISPATCH` 宏注册 CPU 内核调度点

- **设计特点**:
  - 标记为非确定性操作（重复索引时行为不确定）
  - 分离 CPU 特定实现，方便 CUDA/其他后端扩展
  - 遵循 ATen 的 out-variant 编程模式
