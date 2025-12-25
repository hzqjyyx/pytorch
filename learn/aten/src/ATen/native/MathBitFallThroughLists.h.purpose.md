这个文件定义了两个宏，用于注册 PyTorch 张量操作的 fallthrough 实现：

**TORCH_VIEW_FNS(m)**
- 注册视图操作和就地修改版本
- 包含形状变换：reshape、view、squeeze、unsqueeze、transpose、permute、expand 等
- 包含切片分割：narrow、select、split、chunk、unbind 等
- 包含轴操作：movedim、swapaxes、swapdims、hsplit、vsplit、dsplit 等
- 包含复数操作：conj、real、imag、view_as_real
- 包含其他视图：as_strided、detach、diagonal、unfold、unflatten、alias、resize 等

**TENSOR_UTILITIES_AND_CONSTRUCTORS(m)**
- 注册张量工具函数和构造函数
- 张量创建：empty_like、empty、empty_strided、full_like
- 张量属性查询：size、stride、is_complex、is_floating_point
- 其他工具：requires_grad_

**TORCH_VIEW_FNS_NATIVE_FN_REGISTRATION(m)**
- 注册原生函数版本
- as_strided 和 view 的 fallthrough 实现

**关键特点：**
- 使用 `makeFallthrough()` 创建默认实现
- 宏定义便于在多个地方重用同一组注册代码
- 这些都是零复制或低成本操作，不修改张量数据本身
