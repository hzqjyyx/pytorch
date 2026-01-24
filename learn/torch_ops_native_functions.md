# PyTorch 算子系统：从定义到实现

本文档解释 PyTorch 算子系统的核心概念，帮助读者理解如何从 `native_functions.yaml` 定义追踪到具体实现代码。

---

## 1. native_functions.yaml：算子的契约

### 1.1 本质

`native_functions.yaml` 是 PyTorch 算子的**声明式定义**。每个条目描述一个算子的：
- **签名**：输入输出类型
- **变体**：function/method/inplace/out
- **分发规则**：不同后端用什么实现

```yaml
- func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: add_out           # 通用实现
    SparseCPU: add_out_sparse_cpu  # 稀疏张量特化
    MkldnnCPU: mkldnn_add_out      # Intel MKL-DNN 优化
    MPS: add_out_mps               # Apple Metal 后端
```

### 1.2 关键字段

| 字段 | 含义 |
|------|------|
| `func` | 函数签名，`.out` / `.Tensor` 是 overload 名 |
| `structured: True` | 使用 structured kernel（meta/impl 分离） |
| `structured_delegate: xxx.out` | 委托给另一个 structured kernel |
| `dispatch` | 后端 → 实现函数的映射 |
| `variants` | 生成 function 和/或 method 版本 |

---

## 2. 如何找到算子的实现代码

### 2.1 查找流程

```
Step 1: 在 native_functions.yaml 中找到算子定义
        ↓
Step 2: 查看 dispatch 字段，找到目标后端的函数名
        ↓
Step 3: 根据函数名和后端，定位实现文件
```

### 2.2 文件位置规律

| 后端 | 典型位置 | 命名规律 |
|------|---------|---------|
| **通用** | `aten/src/ATen/native/{Op}Ops.cpp` | `TORCH_IMPL_FUNC(xxx_out)` |
| **CPU** | `aten/src/ATen/native/cpu/{Op}Kernel.cpp` | `xxx_kernel` |
| **CUDA** | `aten/src/ATen/native/cuda/{Op}Kernel.cu` | `xxx_kernel_cuda` |
| **MPS** | `aten/src/ATen/native/mps/operations/` | `xxx_out_mps` |
| **Sparse** | `aten/src/ATen/native/sparse/` | `xxx_sparse` |
| **MKL-DNN** | `aten/src/ATen/native/mkldnn/` | `mkldnn_xxx` |

### 2.3 完整示例：追踪 mul 算子

**Step 1: YAML 定义** (`native_functions.yaml:4243`)
```yaml
- func: mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  dispatch:
    CPU, CUDA: mul_out
    MPS: mul_out_mps
    SparseCPU: mul_out_sparse_cpu
```

**Step 2: 通用实现** (`aten/src/ATen/native/BinaryOps.cpp:440`)
```cpp
TORCH_IMPL_FUNC(mul_out)(const Tensor& self, const Tensor& other, const Tensor& result) {
  mul_stub(device_type(), *this);  // 调用 stub 进行二次分发
}
```

**Step 3: CPU 实现** (`aten/src/ATen/native/cpu/BinaryOpsKernel.cpp:126`)
```cpp
void mul_kernel(TensorIteratorBase& iter) {
  // 实际的乘法计算逻辑
}
REGISTER_DISPATCH(mul_stub, &mul_kernel);
```

**Step 4: CUDA 实现** (`aten/src/ATen/native/cuda/BinaryMulKernel.cu:1`)
```cpp
void mul_kernel_cuda(TensorIteratorBase& iter) {
  // CUDA 乘法计算逻辑
}
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda);
```

### 2.4 Stub 机制

Stub 是一个函数指针表，允许运行时根据设备类型选择实现：

```cpp
// 声明 (BinaryOps.h)
DECLARE_DISPATCH(structured_binary_fn, mul_stub);

// 定义 (BinaryOps.cpp)
DEFINE_DISPATCH(mul_stub);

// 注册 (各后端的 Kernel 文件)
REGISTER_DISPATCH(mul_stub, &mul_kernel);      // CPU
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda); // CUDA
```

调用 `mul_stub(device_type(), *this)` 时，根据 `device_type()` 选择对应的实现。

---

## 3. Structured Kernel：Meta/Impl 分离

### 3.1 核心概念

Structured kernel 将算子分为两部分：

| 部分 | 职责 | 宏 |
|------|------|-----|
| **Meta** | 计算输出的 shape、dtype、strides | `TORCH_META_FUNC` |
| **Impl** | 执行实际计算 | `TORCH_IMPL_FUNC` |

### 3.2 为什么要分离？

1. **Shape 推导复用**：Meta 函数在所有后端共享
2. **符号执行支持**：可以只跑 Meta 来推导 shape，不分配内存
3. **一致性保证**：所有后端的 shape 检查逻辑完全相同

### 3.3 示例：threshold 算子

```cpp
// Meta 函数：只计算 shape（Activation.cpp:86）
TORCH_META_FUNC(threshold)(const Tensor& self, const Scalar& threshold, const Scalar& value) {
  build(TensorIteratorConfig()
    .add_output(maybe_get_output())
    .add_const_input(self)
    .add_const_input(self));
}

// Impl 函数：实际计算（Activation.cpp:684）
TORCH_IMPL_FUNC(threshold_out)(const Tensor& self, const Scalar& threshold,
                               const Scalar& value, const Tensor& result) {
  threshold_stub(device_type(), *this, threshold, value);
}
```

### 3.4 structured_delegate 机制

```yaml
# add 的三件套定义
- func: add.Tensor(...)       # functional 变体
  structured_delegate: add.out

- func: add_(...)             # inplace 变体
  structured_delegate: add.out

- func: add.out(...)          # out 变体（核心实现）
  structured: True
```

**委托关系**：
- `add.Tensor` 和 `add_` 都委托给 `add.out`
- 代码生成时，会自动生成调用 structured kernel 的包装代码
- 实际只需实现一次 `add.out` 的 meta 和 impl

---

## 4. 相关算子的联系

### 4.1 三件套模式

大多数算子都有三个变体：

| 变体 | 签名特征 | 行为 |
|------|---------|------|
| **Functional** | `add(Tensor, Tensor) -> Tensor` | 分配新内存，返回新张量 |
| **Inplace** | `add_(Tensor!, Tensor) -> Tensor!` | 修改第一个参数 |
| **Out** | `add.out(..., Tensor! out)` | 写入预分配的输出 |

### 4.2 实现共享

```
add.Tensor ──┐
             ├──→ add.out (structured kernel)
add_       ──┘         ↓
                   meta(): 计算 shape
                       ↓
                   impl(): 调用 add_stub
                       ↓
                   ┌───┴───┐
                   ↓       ↓
                CPU      CUDA
```

### 4.3 生成的调用流程

当用户调用 `torch.add(a, b)` 时：

```
1. torch.add(a, b)
   ↓
2. Dispatcher 查找 add.Tensor 的实现
   ↓
3. 发现 structured_delegate: add.out
   ↓
4. 创建 structured kernel 对象
   ↓
5. 调用 meta() - 计算输出 shape，分配内存
   ↓
6. 调用 impl() - 执行实际计算
   ↓
7. 返回结果
```

### 4.4 特殊情况的独立实现

有些后端需要完全不同的实现，直接在 dispatch 表中指定：

```yaml
- func: add.Tensor(...)
  structured_delegate: add.out
  dispatch:
    SparseCPU, SparseCUDA: add_sparse     # 稀疏张量独立实现
    MkldnnCPU: mkldnn_add                  # MKL-DNN 独立实现
    NestedTensorCPU: NestedTensor_add     # 嵌套张量独立实现
```

这些函数不走 structured kernel 路径，有自己的完整实现。

---

## 5. 二次 Dispatch：以 matmul 为例

### 5.1 概念

"二次 dispatch" 指的是：第一次 dispatch 到达某个后端（如 CUDA）后，该后端内部再次根据条件选择不同的实现。

### 5.2 matmul 的分发链

```
matmul(A, B)
    ↓
[第一层：维度分析 - 运行时]
    ├── 1D @ 1D → dot()
    ├── 2D @ 1D → mv()
    ├── 1D @ 2D → mm().squeeze()
    ├── 2D @ 2D → mm()
    └── 高维    → bmm() 或 折叠后调用 mm()

mm_out_cuda()
    ↓
[第二层：实现选择 - 运行时]
    ├── useLtInterface=true  → gemm_and_bias (cublasLt)
    └── useLtInterface=false → gemm<T>()
                                   ↓
[第三层：后端选择 - 运行时/全局设置]
                              ├── Cublaslt → cublasLtMatmul()
                              ├── Ck (ROCm) → gemm_internal_ck()
                              └── Cublas   → cublasSgemm()
```

### 5.3 分发决策因素

| 层级 | 决策因素 | 类型 |
|------|---------|------|
| 维度分析 | 输入张量的 ndim | 运行时 |
| cublasLt 选择 | dtype、contiguity、矩阵大小 | 运行时 |
| 后端选择 | `globalContext().blasPreferredBackend()` | 全局设置 |
| 数据类型 | `AT_DISPATCH_FLOATING_TYPES_AND2(...)` | 编译期模板 |

### 5.4 关键代码位置

| 组件 | 文件 | 说明 |
|------|------|------|
| matmul 顶层 | `LinearAlgebra.cpp:2002` | 维度分析和路由 |
| mm CUDA 实现 | `cuda/Blas.cpp:668` | 调用 addmm_out_cuda_impl |
| cublasLt 决策 | `cuda/Blas.cpp:336` | useLtInterface 条件判断 |
| gemm 后端选择 | `CUDABlas.cpp:1054` | BlasBackend 分发 |

---

## 6. Composite Ops：组合实现

### 6.1 概念

Composite op 不是独立实现的，而是通过调用其他算子组合而成。

### 6.2 两种类型

| 类型 | 含义 | 例子 |
|------|------|------|
| `CompositeExplicitAutograd` | 显式定义梯度规则 | `add.Scalar` |
| `CompositeImplicitAutograd` | 自动推导梯度 | `broadcast_to`, `chunk` |

### 6.3 示例

```yaml
# add.Scalar 是 Composite，调用 add.Tensor
- func: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
  dispatch:
    CompositeExplicitAutograd: add
```

```cpp
// 实现（BinaryOps.cpp:1159）
Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::add(self, wrapped_scalar_tensor(other), alpha);
  // 包装成 Tensor 后调用 add.Tensor
}
```

### 6.4 Fallback 机制

当某个 dispatch key 没有特定实现时，会 fallback 到 Composite：

```
调用 my_op(tensor)
    ↓
检查 CUDA 实现 → 没有
    ↓
检查 CompositeExplicitAutograd → 有
    ↓
执行 Composite 实现（调用其他基础算子）
    ↓
这些基础算子各自 dispatch 到 CUDA
```

---

## 7. 快速查找指南

### 给定算子名，如何找到实现？

```
1. grep "func: {op_name}" native_functions.yaml
   → 找到定义和 dispatch 表

2. 如果有 structured_delegate
   → 追踪到 .out 变体

3. 查看 dispatch 字段
   → 记录各后端的函数名

4. 搜索函数名
   - 通用：grep "TORCH_IMPL_FUNC({func_name})" aten/src/ATen/native/
   - CPU：  grep "{func_name}" aten/src/ATen/native/cpu/
   - CUDA： grep "{func_name}" aten/src/ATen/native/cuda/
```

### 常用搜索命令

```bash
# 找 add 的所有定义
grep -n "func: add" aten/src/ATen/native/native_functions.yaml

# 找 add_out 的实现
grep -rn "TORCH_IMPL_FUNC(add_out)" aten/src/ATen/native/

# 找 add_stub 的注册
grep -rn "REGISTER_DISPATCH(add_stub" aten/src/ATen/native/
```

---

## 8. 总结

| 概念 | 一句话解释 |
|------|-----------|
| **native_functions.yaml** | 算子的声明式定义，描述签名和分发规则 |
| **structured kernel** | meta/impl 分离，让 shape 推导和计算解耦 |
| **structured_delegate** | 让 functional/inplace 变体复用 out 变体的实现 |
| **stub 机制** | 函数指针表，允许运行时选择后端实现 |
| **二次 dispatch** | 后端内部根据条件再次选择不同实现 |
| **composite op** | 通过组合其他算子实现，无独立 kernel |

---

## 附录：关键文件索引

| 类别 | 路径 |
|------|------|
| 算子定义 | `aten/src/ATen/native/native_functions.yaml` |
| 二元算子 | `aten/src/ATen/native/BinaryOps.cpp` |
| CPU 内核 | `aten/src/ATen/native/cpu/BinaryOpsKernel.cpp` |
| CUDA 内核 | `aten/src/ATen/native/cuda/Binary*.cu` |
| 线性代数 | `aten/src/ATen/native/LinearAlgebra.cpp` |
| CUDA BLAS | `aten/src/ATen/cuda/CUDABlas.cpp` |
| 代码生成 | `torchgen/gen.py`, `torchgen/model.py` |
