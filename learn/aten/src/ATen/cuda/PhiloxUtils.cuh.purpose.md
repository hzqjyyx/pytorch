- **Philox RNG 状态管理** - 为 CUDA 核心提供 Philox 伪随机数生成器的状态管理工具

- **CUDA Graph 兼容** - 支持 CUDA Graph 捕获模式和普通 Eager 模式下的 RNG 状态访问

- **内核内解包函数** - 提供 `__device__` 内联函数从 `PhiloxCudaState` 实例中提取伪随机数的种子和偏移量

- **双模式支持** - 根据是否启用 Graph 捕获，灵活处理指针间接寻址或直接值访问

- **cuDNN 集成** - 提供 `unpack_cudnn` 全局核函数和包装器用于从 PhiloxCudaState 提取种子和偏移

- **JIT 代码生成兼容** - 使用原始定义文件设计以便 JIT 代码生成器易于复制
