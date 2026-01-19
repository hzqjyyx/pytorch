这个文件实现了 PyTorch 的 **Functionalization Pass** 的回退内核（fallback kernel），主要目的是将带有副作用的操作（mutations）转换为函数式的操作。

## 核心机制

**functionalizeFallback** (31-121行)
- 作为通用回退处理器，处理所有没有专门实现的算子
- 检查算子是否有别名注解（alias annotations），如果有则报错，因为只支持无别名输出的算子
- 处理流程：
  1. 遍历输入参数，如果是 FunctionalTensor，则同步（sync）并解包（unwrap）
  2. 跳过 Functionalize key，调用原始算子
  3. 如果输入有 FunctionalTensor 或者是工厂函数，则将输出包装成 FunctionalTensor

## 特殊算子实现

**resize__functionalization** (128-186行)
- `resize_()` 比较特殊，因为它既可能是 mutation 也可能是 view：
  - 扩大尺寸时：作为 mutation，需要替换底层 storage
  - 缩小尺寸时：作为 view，用 `as_strided` 模拟切片行为

**_unsafe_view_functionalize** (289-322行)
- 手动处理 `_unsafe_view` 的别名关系
- 创建带有 view metadata 的 FunctionalTensor，记录前向和反向的 view 操作
- 推断并设置正确的 sizes 和 strides

**set__functionalize** (324-347行)
- 处理 `set_()` 操作，将一个 tensor 的内容设置为另一个 tensor
- 检查两个 tensor 都必须是 FunctionalTensor
- 调用内部的 `set__impl` 来更新引用关系

**lift 系列函数** (189-226行)
- `lift_functionalize`: 将普通 tensor 提升为 FunctionalTensor
- `lift_fresh_functionalize`: 如果已经是 FunctionalTensor，返回 view；否则提升
- `lift_fresh_functionalize_copy`: 如果已经是 FunctionalTensor，返回 clone；否则提升

**_to_copy_functionalize** (236-268行)
- 处理设备间的 tensor 拷贝
- 特殊逻辑：在 lazy backend（如 XLA/LTC）上，如果拷贝到非 functionalization 设备，则结束 functionalization pass

## 注册机制

- 349-351行：为所有算子注册通用的 fallback 处理器
- 353-363行：为特定的 aten 算子注册专门的实现

---

**ROCm/Backward 相关：**
- 无 ROCm 特定代码
- 无显式的 Backward 实现（functionalization 主要在前向传播中工作，autograd 会在更高层处理）
