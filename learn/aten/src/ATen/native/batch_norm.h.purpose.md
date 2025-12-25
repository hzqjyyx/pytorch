- **批量归一化（Batch Normalization）的核心声明文件**
  - 定义了三个函数指针类型，用于调度不同后端的实现
  
- **函数指针类型定义**
  - `batch_norm_fn`: 主要的批量归一化前向计算
  - `batch_norm_collect_stats_fn`: 用于收集统计信息（均值、方差）
  - `batch_norm_backward_fn`: 反向传播计算
  
- **调度声明**
  - `DECLARE_DISPATCH` 宏为上述三种操作声明 CPU 后端的调度桩
  - 允许在运行时根据输入特性选择优化的实现
  
- **辅助工具函数**
  - `conditional_accessor_1d()`: 处理可能未定义的张量，返回安全的 1D 访问器
  - `conditional_data_ptr()`: 获取张量数据指针，处理常量和非常量情况，支持张量未定义的情况
  
- **设计模式**
  - 使用 DispatchStub 机制实现后端抽象
  - 通过条件访问器优雅地处理可选张量参数
