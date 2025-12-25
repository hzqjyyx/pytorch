这个文件提供了 Tensor 方法的本地函数桩（native function stubs），用于动态分派（dynamic dispatch）。文件中的每个函数都是对 Tensor 类成员方法的简单包装，将调用重定向到对应的成员方法实现。

**主要功能：**

- **梯度相关**：`_backward()` - 触发反向传播，支持梯度累积和图构建选项
- **数据管理**：`set_data()` / `data()` - 设置和获取张量底层数据
- **自动求导配置**：`requires_grad_()` - 设置张量是否需要梯度
- **梯度保留**：`retain_grad()` / `retains_grad()` - 管理叶子节点梯度保留策略
- **张量属性查询**：`is_leaf()` / `output_nr()` / `_version()` - 查询张量的自动求导属性和版本信息
- **推理模式支持**：`_fw_primal()` - 在推理模式下获取张量的前向原始值（带断言检查）

**设计模式：**

- 头条件编译分支：`AT_PER_OPERATOR_HEADERS` 控制是否使用细粒度操作头文件
- 所有函数都是轻量级包装，实际逻辑在 `TensorBody.h` 中的 Tensor 成员方法
- 通过 namespace `at::native` 暴露给动态分派系统
