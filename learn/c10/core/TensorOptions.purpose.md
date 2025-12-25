## TensorOptions 核心功能

`TensorOptions` 是一个构建器类，用于封装创建 Tensor 时的配置参数。它模拟 Python 中关键字参数的功能，例如 `torch.zeros(2, 3, dtype=torch.int32)`。

### 设计动机

C++ 不支持关键字参数，因此需要一个类来充当"参数字典"。`TensorOptions` 允许链式调用来指定参数：

```cpp
at::zeros({2,2}, at::device(at::kCUDA).dtype(at::kLong));
at::zeros({2,2}, at::kCUDA);  // 隐式转换
```

### 核心属性

所有属性都是可选的（optional），使用 `has_*` 布尔标志跟踪是否被设置：

- `dtype_`: 数据类型（默认 float）
- `device_`: 设备（默认 CPU）
- `layout_`: 内存布局（默认 Strided）
- `requires_grad_`: 是否需要梯度（默认 false）
- `pinned_memory_`: 是否使用固定内存（默认 false）
- `memory_format_`: 内存格式（Contiguous 等）

### API 设计

**两类方法：**

1. **Getter 方法**：返回属性值，未设置时返回默认值
   - `device()` → 返回设备（默认 CPU）
   - `has_device()` → 是否显式设置
   - `device_opt()` → 返回 `std::optional<Device>`

2. **Setter 方法**：返回新的 `TensorOptions` 对象（不可变风格）
   ```cpp
   [[nodiscard]] TensorOptions device(std::optional<Device> device) const noexcept;
   ```

### 隐式构造函数

为了支持简洁的 API，提供多个隐式构造函数：

```cpp
TensorOptions(Layout layout)
TensorOptions(ScalarType dtype)
TensorOptions(Device device)
TensorOptions(MemoryFormat memory_format)
```

**特殊处理 Device 构造**：使用模板避免与拷贝构造函数的歧义，因为 `Device` 自身也有隐式构造函数。

### 关键操作

**merge_in()**：右偏合并两个 `TensorOptions`
- 用于函数如 `Tensor.new_empty()` 时覆盖默认选项
- 注意：设备合并不遵循索引继承（如 `device({kCUDA, 1}).merge_in(kCUDA)` 结果是 `kCUDA` 而非 `{kCUDA, 1}`）

**computeDispatchKey()**：根据 dtype/layout/device 计算调度键
- 处理密集张量、稀疏张量、量化张量等不同组合
- 映射到具体后端实现（CPU/CUDA/Sparse/Quantized 等）

### 内存优化

使用位域压缩布尔标志，确保整个对象不超过 128 位（两个机器字）：

```cpp
bool requires_grad_ : 1;
bool has_device_ : 1;
// ... 其他标志
```

### 流式输出

`operator<<` 实现了诊断输出，显示所有属性及其是否为默认值：

```
TensorOptions(dtype=float (default), device=CPU (default), layout=Strided (default), ...)
```

### 辅助工具

**默认值函数**：
- `dtype_or_default()` → 未设置时返回全局默认 dtype
- `device_or_default()` → 未设置时返回 CPU
- `layout_or_default()` → 未设置时返回 Strided

**便利构造函数**：
```cpp
at::dtype(at::kInt)
at::device(at::kCUDA)
at::requires_grad()
```

### 关键约束

- `memory_format()` 故意不提供默认值 getter，因为不同函数的默认行为不同
- Quantized 后端不支持 `at::empty()`，需要专门的量化操作符
- 稀疏压缩格式（CSR/CSC/BSR/BSC）共享部分调度逻辑

---

**其他简要说明：**
- 支持多种稀疏布局：Sparse, SparseCsr, SparseCsc, SparseBsr, SparseBsc
- `dispatchKeyToLayout/DeviceType/TensorOptions` 提供反向映射
- `type_equal()` 用于兼容遗留的 `tensor.type()` 比较
- 代码中包含对过时 Caffe2 设备类型（MKLDNN/OPENGL/OPENCL/IDEEP）的断言保护
