这个文件提供了用于 functorch 批处理规则的模板元编程工具，主要用于从算子签名自动推导批处理规则的签名。

## 核心功能

**类型列表操作**
- 定义了 `typelist`、`head_t`、`concat_t` 等基础类型列表操作
- 实现了 `tail` 操作，用于获取类型列表除第一个元素外的剩余部分

**批次维度移除逻辑**
`RemoveBatchDimAfterTensor` 是核心转换器，它递归遍历类型列表，识别并移除 Tensor 参数后面的 `std::optional<int64_t>` 批次维度参数。

支持的 Tensor 类型模式：
- `Tensor` / `const Tensor&` / `Tensor&`
- `std::optional<Tensor>` 及其引用变体
- `std::vector<Tensor>`

当检测到这些 Tensor 类型后跟 `std::optional<int64_t>` 时，会跳过该批次维度参数，只保留 Tensor 参数本身。

**函数类型构建**
- `BuildFunctionHelper` / `BuildFunction`: 从返回类型和参数类型列表构建函数类型
- `UnpackSingleItemTuple`: 将单元素 tuple 解包为其内部类型

**批处理规则到算子类型的转换**
`ToOperatorType` 模板实现完整转换流程：
1. 提取批处理规则的返回类型和参数类型
2. 对参数类型应用 `remove_batch_dim_after_tensor_t`，移除批次维度
3. 对返回类型（通常是 tuple）应用相同转换并解包
4. 重新构建算子函数类型

## 使用场景

这些元编程工具支持 vmap plumbing 机制，允许：
- 从批处理规则签名自动推导原始算子签名
- 在编译时验证批处理规则与算子的类型匹配
- 减少手动编写类型转换代码的需求

---

**忽略的内容：**
- ROCm 相关：无
- Backward 相关：无
