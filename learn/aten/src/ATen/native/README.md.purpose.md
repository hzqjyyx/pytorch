# ATen Native Functions 主要功能

## 核心机制

ATen "native" functions 是向 ATen 添加算子和函数的现代机制。这些函数在 `native_functions.yaml` 中声明，实现代码位于当前目录的 cpp 文件中。

## 函数注册 (`native_functions.yaml`)

### 基本格式
```yaml
- func: func_name(ArgType arg0[=default], ...) -> Return
  variants: function, method
  dispatch:
    CPU: func_cpu
    CUDA: func_cuda
```

### `func` 签名定义

**参数类型：**
- `Tensor` / `Tensor?` - 张量（可选）
- `Tensor[]` - 张量数组
- `int[]` / `int` / `float` / `bool` / `str` - 基础类型
- `Scalar` - 可绑定任意数值类型
- `Generator?` - 随机数生成器
- `*` - Python 中标记后续参数必须为关键字参数

**别名和变异注解：**
- `Tensor(a)` - 可能别名相同数据
- `Tensor(a!)` - 可能被写入（变异）
- `Tensor(a! -> a|b)` - 写入后同时属于集合 a 和 b
- 用于标识 inplace 操作（如 `abs_`）和 out 参数（如 `abs(..., *, Tensor(a!) out)`）

**返回值：**
- 单个：`Tensor` / `Tensor[]`
- 元组：`(Tensor, Tensor, ...)`
- 可指定返回参数名用于 derivatives.yaml 和 Python 访问

**重载（Overloads）：**
- 使用点号分隔：`func_name.overload_name`
- 重载名必须在同名函数集合中唯一
- 向后兼容要求，不可更改

### `variants` - 生成变体

```yaml
variants: function, method
```

- `function`: 生成命名空间函数 `at::foo()`
- `method`: 生成张量方法 `t.foo()`（需要有 `Tensor self` 参数）
- 默认只生成 function 变体

### `dispatch` - 后端分发

指定不同后端的实现函数：

```yaml
dispatch:
  CPU: func_cpu
  CUDA: func_cuda
  CompositeImplicitAutograd: func  # 通用实现，自动支持 autograd
  CompositeExplicitAutograd: func  # 通用实现，需手动定义梯度
```

**三种特殊后端：**

1. **CompositeImplicitAutograd**（默认）
   - 适用所有后端的实现
   - 自动支持 autograd（因调用的操作支持 autograd）
   - 无 dispatch 段时的默认选择

2. **CompositeExplicitAutograd**
   - 适用所有后端
   - 需在 `derivatives.yaml` 中显式定义反向传播

3. **CompositeExplicitAutogradNonFunctional**
   - 非别名算子内部调用别名算子时使用
   - 某些后端（如 XLA）不希望将非别名操作分解为别名操作

**选择 dispatch 关键字的决策流程：**

1. 推理阶段：是否支持所有后端？
   - 否 → 枚举后端：`CPU: kernel_cpu, CUDA: kernel_cuda`
   - 是 → 进入步骤 2

2. 训练阶段：是否自动支持 autograd？
   - 是 → 省略 dispatch 段（默认 CompositeImplicitAutograd）
   - 是，但想自定义梯度 → `CompositeExplicitAutograd: kernel`
   - 否 → `CompositeExplicitAutograd: kernel`

### 其他属性

**`device_guard`**: `False` - 禁用自动设备切换

**`device_check`**: `NoCheck` - 禁用设备一致性检查

**`manual_kernel_registration`**: `True` - 手动注册到 dispatcher

**`use_const_ref_for_mutable_tensors`**: `True` - 可变张量也用 `const Tensor&`

**`autogen`**: 自动生成函数变体
```yaml
- func: my_op_(Tensor(a!) self) -> Tensor(a!)
  autogen: my_op, my_op.out  # 生成 functional 和 out= 变体
```

**`python_module`**: 指定 Python 模块（`nn`, `fft`, `linalg`, `sparse`, `special`, `nested`）

**命名空间**：支持自定义命名空间
```yaml
- func: custom::my_op(...) -> Tensor
```

## C++ 实现规范

### Composite Compliance 约束

CompositeImplicitAutograd 函数必须适用于所有后端/子类，**禁止**：
- 调用 `resize_` 或等价操作
- 调用 `out=` 操作
- 直接修改 TensorImpl 元数据
- 访问 `data_ptr` 或 `item`
- 避免使用 `copy_`（除非无法避免）

### Undefined Tensor 约定

- 输入：默认必须已定义，除非声明 `Tensor?`
- 输出：
  - backward 函数可根据 `output_mask[i]` 返回 undefined
  - 其他情况禁止返回 undefined（应返回零大小张量或零填充张量）

---

**简要列出的内容：**

• **ROCm**: 文档中未特别提及 ROCm，后端分发机制同样适用

• **Backward 相关**: 
  - 需在 `derivatives.yaml` 中关联 `foo` 和 `foo_backward`
  - backward 函数可根据 `output_mask` 返回 undefined tensor
  - 非自动可微函数需显式定义梯度公式
  - 使用 `CompositeExplicitAutograd` 需配合 `derivatives.yaml` 支持训练
