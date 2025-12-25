**TensorWrapper 的主要功能**

TensorWrapper 是 functorch 库中用于梯度变换（grad、vjp、jvp）的张量包装子类，类似于 vmap 中的 BatchedTensor。其核心设计目的是支持嵌套的梯度计算，例如 `grad(grad(torch.sin))(x)`，通过层层包装使每个 TensorWrapper 持有独立的 AutogradMeta，从而参与独立的自动求导图。

**TensorWrapper 结构**

- 继承自 `c10::TensorImpl`，包含：
  - `value_`：被包装的底层张量
  - `level_`：该包装器所在的转换层级
  - `is_immutable_`：标记张量是否来自别名不可变张量的操作
  - `is_alive_`：shared_ptr<bool>，追踪对应 Grad Interpreter 是否仍活跃

**核心方法**

- `refreshMetadata()`：同步底层张量的大小、步长和存储偏移到包装器元数据
- `shallow_copy_and_detach()`：创建浅拷贝并分离（支持 Autograd 需求）
- `level()`：返回当前层级（若活跃）或空值（若已死亡）
- `is_alive()`：检查对应解释器是否仍活跃

**死亡包装器行为**

当退出变换层级时，包装器标记为"死亡"但仍保留 Autograd 元数据。死亡包装器通过后备内核 `dead_tensor_wrapper_fallback()` 自动解包其底层值并重新分发，实现对常规张量的行为转换。

**API 接口**

- `makeTensorWrapper(tensor, level/interpreter, is_immutable)`：创建包装张量
- `maybeGetTensorWrapper(tensor)`：安全地提取 TensorWrapper 指针（若不是包装张量返回 nullptr）
- `dumpTensor()`/`dumpTensorCout()`：调试输出张量包装层次结构

**要点总结**

- 支持嵌套梯度计算的张量子类包装
- 通过 DispatchKey::FuncTorchGradWrapper 标识
- 活跃期间维持独立的 AutogradMeta；死亡后自动解包转发
- 与 VariableType Autograd 内核兼容，最小化修改
- 防止直接变异（通过 `.data` 在 functorch 转换内抛出错误）
