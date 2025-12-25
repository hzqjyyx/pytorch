**PhiloxCudaState.h 主要功能：**

- **定义 RNG 状态结构**：`PhiloxCudaState` 结构体用于存储和传递 CUDA 随机数生成器的状态信息
- **支持两种初始化模式**：
  - 普通模式：直接使用 seed 和 offset 值
  - 图捕获模式：使用指针存储 seed 和 offset，用于 CUDA Graph 场景
- **灵活的数据存储**：使用 `union Payload` 实现值和指针的二元性（支持图捕获时的动态更新）
- **作为核函数参数**：设计为可以直接作为 CUDA 核函数的参数传递
- **头文件导出层**：`PhiloxCudaState.h` 作为公共接口，而实现细节在 `.cuh` 文件中，便于 JIT 代码生成器复制使用
