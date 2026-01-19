# TensorBody.h 主要功能分析

这是 PyTorch 的核心 Tensor 类模板定义文件，定义了 `at::Tensor` 类的完整接口。

## 核心职责

**类继承关系**
- `Tensor` 继承自 `TensorBase`
- 使用 intrusive_ptr 管理 `TensorImpl` 的引用计数
- 类似 `boost::intrusive_ptr` 的智能指针语义

**引用计数语义** (75-91行)
```cpp
void func(Tensor a) {
  Tensor b = a;  // 共享 TensorImpl，引用计数+1
  ...
}  // b 析构时引用计数-1
```

## 主要功能模块

### 1. 构造与赋值

**构造函数** (103-114行)
- 默认构造、拷贝构造、移动构造
- 从 `TensorImpl` 指针构造
- 从 `TensorBase` 隐式/显式转换

**赋值运算符重载** (198-222行)
- 使用 ref-qualifier 区分左值/右值语义
- 左值赋值 (`&`): 共享 TensorImpl，不拷贝数据
- 右值赋值 (`&&`): 执行数据拷贝
  ```cpp
  Tensor x = y;      // 共享数据
  x[1] = 3;          // 右值赋值，拷贝数据到 x[1]
  ```

### 2. 设备转换方法 (333-360行)

```cpp
Tensor cpu() const;     // 转到 CPU
Tensor cuda() const;    // 转到 CUDA
Tensor hip() const;     // 转到 HIP
Tensor vulkan() const;  // 转到 Vulkan
Tensor metal() const;   // 转到 Metal
Tensor meta() const;    // 转到 Meta 设备
```

### 3. 运算符重载 (263-323行)

**算术运算符**
- `operator~()`: 按位取反
- `operator-()`: 取负
- `operator+=, -=, *=, /=`: 复合赋值
- `operator&=, |=, ^=`: 按位运算

**索引运算符** (302-323行)
- `operator[](Scalar)`: 标量索引
- `operator[](Tensor)`: 张量索引
- `operator[](int64_t)`: 整数索引，调用 `select(0, index)`

**高级索引** (325-331行)
```cpp
Tensor index(ArrayRef<TensorIndex> indices);
Tensor& index_put_(ArrayRef<TensorIndex> indices, const Tensor& rhs);
```

### 4. 内存布局与连续性

**连续性检查** (123-159行)
- `contiguous(MemoryFormat)`: 返回连续张量
- `expect_contiguous(MemoryFormat)`: 性能优化版本
  - 如果已连续：返回 borrowed reference（不增加引用计数）
  - 如果不连续：返回新的连续副本

**共轭操作** (127-142行)
- 对于复数张量，根据布局选择物理共轭或逻辑共轭
- 稀疏张量使用 `conj_physical()`，其他使用 `_conj()`

### 5. Autograd API (362-634行)

**梯度属性**
- `is_leaf()`: 判断是否为叶子节点
- `grad()`: 获取梯度（只读）
- `mutable_grad()`: 获取可变梯度引用
- `requires_grad_(bool)`: 设置是否需要梯度

**梯度计算** (436-447行)
```cpp
void backward(
  const Tensor& gradient={},
  std::optional<bool> retain_graph=std::nullopt,
  bool create_graph=false,
  std::optional<TensorList> inputs=std::nullopt
);
```

**图操作**
- `detach()`: 从计算图分离（返回新张量）
- `detach_()`: 原地分离
- `retain_grad()`: 非叶子节点保留梯度

**Hook 机制** (587-620行)
```cpp
template <typename T>
unsigned register_hook(T&& hook);  // 注册反向传播钩子
```

### 6. Forward Mode AD (499-513行)

```cpp
const Tensor& _fw_grad(uint64_t level);
void _set_fw_grad(const TensorBase& new_grad, uint64_t level, bool is_inplace_op);
```

### 7. 数据访问

**原始数据指针** (245-249行)
```cpp
template<typename T>
T* data() const;  // 已弃用，使用 data_ptr<T>()
```

**标量提取** (251-252行)
```cpp
template <typename T>
T item() const;  // 提取零维张量的标量值
```

**Accessor** (254-261行)
```cpp
GenericPackedTensorAccessor<T,N,PtrTraits,index_t> packed_accessor();
```

### 8. 类型与设备转换

**类型转换** (231-238行)
```cpp
Tensor toType(ScalarType t);
Tensor toBackend(Backend b);  // 已弃用
```

**灵活转换** (540-545行)
```cpp
Tensor to(TypeMeta type_meta, bool non_blocking, bool copy);
Tensor to(Device device, TypeMeta type_meta, bool non_blocking, bool copy);
```

### 9. 变量数据视图 (560-577行)

- `tensor_data()`: 共享存储但不共享元数据变更
- `variable_data()`: 共享存储但创建新的 autograd 历史

### 10. 模板方法 (547-550行)

```cpp
template <typename F, typename... Args>
decltype(auto) m(F func, Args&&... params) const {
  return func(*this, std::forward<Args>(params)...);
}
```
函数式编程风格的方法调用包装器。

## 特殊设计

**MaybeOwnedTraits** (658-698行)
- 实现零开销的借用语义
- `createBorrow()`: 创建 +0 引用的借用
- `destroyBorrow()`: 不减少引用计数

**ExclusivelyOwnedTraits** (701-734行)
- 独占所有权的类型特征
- 用于移动语义和资源管理

**代码生成占位符** (521, 652行)
```cpp
${tensor_method_declarations}  // 从 native_functions.yaml 生成
${tensor_method_definitions}   // 生成方法定义
```

## 已弃用 API

- `type()`: 使用 `options()` 替代 (224-229行)
- `is_variable()`: 所有张量都是变量 (240-243行)
- `data<T>()`: 使用 `data_ptr<T>()` 替代 (245-249行)
- `packed_accessor()`: 使用 `packed_accessor32/64` (254-261行)

---

**ROCm 相关**: `hip()` 方法用于 AMD GPU 设备转换

**Backward 相关**: 
- `backward()` 方法及其参数 `gradient`, `retain_graph`, `create_graph`, `inputs`
- `_backward()` 内部实现
- `grad()`, `mutable_grad()` 梯度访问
- `register_hook()` 反向传播钩子注册
