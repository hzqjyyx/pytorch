* **文件用途**: 定义嵌套张量（Nested Tensor）在 SDPA（Scaled Dot-Product Attention）操作中的预处理函数接口

* **主要函数**:
  - `sdpa_nested_preprocessing()`: 将嵌套的 query、key、value 张量转换为可与 flash-attention 或 efficient-attention 内核兼容的格式，返回处理后的张量和相关元数据
  - 输出包括转换后的张量、最大序列长度等信息

* **核心作用**: 作为嵌套张量与高效注意力机制之间的适配层，负责数据格式转换和参数提取

* **命名空间**: 位于 `at::native::preprocessing` 下，表示这是 ATen 库中的原生预处理操作

* **头文件性质**: 仅包含函数声明，实现在其他 .cpp 文件中
