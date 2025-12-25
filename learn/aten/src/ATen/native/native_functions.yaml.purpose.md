## native_functions.yaml 主要功能

这是 **PyTorch ATen 库的核心算子声明文件**，定义了所有原生函数的接口规范。它是代码生成的"配置中心"。

### 核心作用

**1. 函数签名定义**
- 声明函数名、参数类型（Tensor, int, float, Scalar等）、返回值
- 支持重载（通过 `.overload_name` 区分）
- 定义参数默认值和可选参数（`?` 标记）

**2. 别名与变异标注（Aliasing & Mutation）**
- `Tensor(a!)` - 可能被写入的张量
- `Tensor(a)` - 只读的视图别名
- `Tensor(a! -> a|b)` - 写入后别名关系变化
- 区分 inplace 操作（`_` 后缀）和 out 变体

**3. 多后端分发配置**
```yaml
dispatch:
  CPU: func_cpu
  CUDA: func_cuda
  MPS: func_mps
  CompositeExplicitAutograd: generic_impl
```
- 指定不同设备后端的具体实现函数
- 支持通用实现（Composite*）和专用优化

**4. 代码生成驱动**
从这个 YAML 自动生成：
- C++ 头文件和函数声明（`at::` 命名空间）
- Tensor 方法绑定（`tensor.foo()`）
- Python 绑定（torch.* API）
- 调度器注册代码
- 函数式变体（通过 `autogen` 关键字）

**5. Autograd 集成信息**
- 标记是否需要手动定义导数（`CompositeExplicitAutograd`）
- 或自动推导梯度（`CompositeImplicitAutograd`）
- 关联 `derivatives.yaml` 中的反向传播规则

### 典型条目结构示例
```yaml
- func: atan(Tensor self) -> Tensor
  structured_delegate: atan.out
  variants: function, method
  dispatch:
    SparseCPU, SparseCUDA: atan_sparse
    CPU, CUDA: atan_out
  tags: [core, pointwise]
```

### 与其他组件协同
- **native/*.cpp** - 实际的 C++ 实现
- **derivatives.yaml** - 反向传播公式
- **torchgen** - 代码生成工具链
- **dispatcher** - 运行时分发机制

---

**简要提及的内容：**
- **ROCm 支持**: `dispatch` 中可指定 `HIP/ROCM` 后端
- **Backward 相关**: 包含大量 `*_backward` 函数定义（如 `binary_cross_entropy_backward`），用于自动微分
