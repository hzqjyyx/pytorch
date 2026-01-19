这是一个模板文件，用于自动生成 PyTorch 的 LazyTensor 非原生 IR 节点。

**主要功能：**

- **模板框架**：提供基础结构来生成 LazyTensor 相关的中间表示（IR）节点代码
- **命名空间管理**：通过 `${namespace_prologue}` 和 `${namespace_epilogue}` 占位符管理代码的命名空间
- **IR 节点生成**：`${non_native_ir_nodes}` 占位符用于插入自动生成的非原生操作的 IR 节点定义
- **头文件包含**：`${lazy_non_native_ir_inc}` 占位符用于包含必要的依赖头文件
- **代码生成工具**：这个文件本身不包含实现，而是被代码生成工具（如 codegen 脚本）处理，用占位符替换生成最终的 C++ 头文件
