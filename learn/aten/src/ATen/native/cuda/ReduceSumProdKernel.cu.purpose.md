**文件主要功能：**

- **Sum 操作**：实现 CUDA 张量求和内核，包括标准求和和 NaN 跳过求和（nansum）
  - 针对 Half 和 BFloat16 类型的特殊优化路径
  - 复数 Half 类型使用 JIT 编译或 GPU Lambda 实现

- **Product 操作**：实现 CUDA 张量乘积内核
  - 布尔类型特殊处理（用 `&&` 替代 `*`）
  - 复数 Half 类型单独实现

- **类型提升机制**：
  - Half → Float 提升（可选输出类型转换）
  - BFloat16 → Float 提升
  - 在单个内核中完成转换和归约

- **分发系统**：
  - `reduce_dispatch` 模板函数处理类型判断和路由
  - 支持所有数值类型、复数、布尔类型
  - 通过 `AT_DISPATCH_*` 宏展开具体类型实现

- **内核注册**：
  - 通过 `REGISTER_DISPATCH` 宏注册到对应的 stub（sum_stub, nansum_stub, prod_stub）
  - 用于 PyTorch 的动态分发系统
