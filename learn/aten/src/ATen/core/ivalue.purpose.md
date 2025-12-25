# IValue 核心功能分析

## 核心概念

IValue (Interpreter Value) 是 PyTorch TorchScript 解释器使用的**tagged union**（带标签联合体），用于统一表示所有 TorchScript 支持的值类型。

- **内存布局**：16字节对象 = 8字节 payload + 8字节 tag
- **设计目标**：在解释器中高效传递和存储不同类型的值

## 支持的类型（Tag枚举）

通过 `TORCH_FORALL_TAGS` 宏定义了27种类型：

**基础类型**：None, Int, Double, Bool, String, ComplexDouble  
**符号类型**：SymInt, SymFloat, SymBool（用于符号计算）  
**张量相关**：Tensor, Storage, Generator, Quantizer  
**容器类型**：Tuple, GenericList, GenericDict  
**设备相关**：Device, Stream  
**特殊类型**：Object（自定义类）, PyObject, Capsule, Enum, Future, Await, RRef, Blob, Uninitialized

## Payload 存储机制

```cpp
union Payload {
    union TriviallyCopyablePayload {
        int64_t as_int;
        double as_double;
        bool as_bool;
        intrusive_ptr_target* as_intrusive_ptr;  // 引用计数指针
        struct { DeviceType type; DeviceIndex index; } as_device;
    } u;
    at::Tensor as_tensor;  // Tensor 单独存储
}
```

**存储策略**：
- 原始类型（int/double/bool/device）直接存储值
- Tensor 使用专用字段存储（优化性能）
- 其他复杂类型通过 `intrusive_ptr` 间接存储
- null 指针统一表示为 `UndefinedTensorImpl::singleton()`

## 核心操作实现

### 1. 类型检查与转换（ivalue.h）

```cpp
bool isTensor() const { return Tag::Tensor == tag; }
at::Tensor toTensor() const& { /* 直接访问 payload.as_tensor */ }

bool isInt() const { return Tag::Int == tag; }
int64_t toInt() const {
    if (isInt()) return payload.u.as_int;
    else if (isSymInt()) return toSymInt().guard_int(...);  // 符号值降级
}
```

### 2. 相等性比较（ivalue.cpp:283-362）

`operator==` 的语义遵循 Python 规则：

- **数值类型**：按值比较（lines 318-330）
- **Tensor**：返回 `BoolTensor`（逐元素比较，line 313）
- **容器类型**：递归比较元素（lines 334-342）
  - List/Tuple：字典序比较
  - Dict：键值对必须完全相等
- **引用类型**：指针相等性（line 352）
- **特殊优化**：身份相等隐含值相等（line 19-24，`_fastEqualsForContainer`）

### 3. 哈希计算（ivalue.cpp:364-414）

```cpp
size_t hash(const IValue& v) {
    switch (v.tag) {
        case Tag::Tensor:
            return c10::get_hash(v.payload.as_tensor.unsafeGetTensorImpl());  // 指针哈希
        case Tag::Int/Bool/Double:
            return c10::get_hash(v.payload.u.as_int/as_bool/as_double);
        case Tag::GenericDict/GenericList/...:
            throw "unhashable type";  // Python 语义
    }
}
```

**不可哈希类型**：List, Dict, Object, Future 等（遵循 Python 约定）

### 4. 序列化表示（ivalue.cpp:576-661）

**repr()**：生成可重建 IValue 的 TorchScript 表达式
```cpp
IValue(26).repr(out)           → "26"
IValue(3.14).repr(out)         → "3.14"
IValue("hello").repr(out)      → '"hello"'
IValue({1,2}).repr(out)        → "annotate(List[int], [1, 2])"  // 空列表需类型注解
```

**operator<<**：调试输出（更宽松的格式）

### 5. 深拷贝（ivalue.cpp:884-975）

```cpp
IValue deepcopy(std::optional<Device> device) const {
    HashIdentityIValueMap memo;  // 防止循环引用
    switch(tag) {
        case Tensor:
            return device ? tensor.to(*device) : tensor.clone();
        case GenericList:
            // 递归拷贝每个元素
        case Object:
            // 优先使用 __getstate__/__setstate__
            // 否则逐属性深拷贝
    }
}
```

支持跨设备拷贝（CPU ↔ GPU）

### 6. 排序比较器（ivalue.cpp:690-778）

```cpp
IValueComparator getLessThanComparator(const IValue& v) {
    if (v.isTensor()) 
        return [](a, b) { return a.toTensor().lt(b.toTensor()).is_nonzero(); };
    if (v.isInt()) 
        return [](a, b) { return a.toInt() < b.toInt(); };
    if (v.isTuple()) {
        // 字典序比较：逐元素递归
    }
    if (v.isObject()) {
        // 查找并调用 __lt__ 方法
    }
}
```

用于 TorchScript 的 `sorted()` 函数

### 7. 访问者模式（ivalue.cpp:157-204, 206-265）

```cpp
void visit(const std::function<bool(const IValue&)>& visitor) const {
    if (visitor(*this)) return;  // 短路
    switch (tag) {
        case Tuple/GenericList:
            for (auto& elem : elements) elem.visit(visitor);
        case GenericDict:
            for (auto& pair : dict) {
                pair.key().visit(visitor);
                pair.value().visit(visitor);
            }
        case Object:
            // 递归访问所有属性
    }
}
```

**应用场景**：
- `getSubValues()`：收集所有子张量（用于别名分析，lines 206-265）
- `overlaps()`：检测两个 IValue 是否共享底层存储（line 267-277）

## 引用计数管理

### intrusive_ptr 优化（ivalue.h:1189-1203, 1298-1322）

```cpp
void destroy() {
    if (isTensor() || isIntrusivePtr()) {
        intrusive_ptr_target* p = isTensor() 
            ? payload.as_tensor.unsafeGetTensorImpl()
            : payload.u.as_intrusive_ptr;
        intrusive_ptr::reclaim(p);
        // 不调用 ~Tensor()（移动后的 Tensor 已是 null 状态）
    }
}
```

**位向量优化**（lines 1310-1322）：
- 用32位整数的每一位表示对应 Tag 是否为 intrusive_ptr
- 避免 switch 语句的边界检查分支

## 特殊类型处理

### 1. 符号值（SymInt/SymFloat/SymBool）

构造时自动区分具体值和符号值（ivalue.h:588-630）：
```cpp
IValue(const SymInt& i) {
    if (auto mi = i.maybe_as_int()) {
        tag = Tag::Int;
        payload.u.as_int = *mi;  // 存储为普通 int
    } else {
        tag = Tag::SymInt;
        payload.u.as_intrusive_ptr = i.toSymNode().release();  // 存储符号节点
    }
}
```

### 2. WeakIValue（ivalue.h:1386-1506）

弱引用版本，用于避免循环引用：
```cpp
IValue lock() const {
    if (Tag::Tensor == tag) {
        auto ip = weak_intrusive_ptr::lock();
        return ip ? IValue(Tensor(ip)) : IValue();  // 返回 None 如果已释放
    }
}
```

### 3. Future/Await（ivalue.cpp:1076-1131, 1133-1241）

异步值容器：
```cpp
// extractStorages：提取所有张量的底层 Storage
std::vector<weak_intrusive_ptr<StorageImpl>> extractStorages(const IValue& value) {
    if (value.isPyObject()) {
        // 通过 pickling 提取张量
    } else {
        value.getSubValues(sub_values);
        // 处理稀疏张量（indices + values 各有一个 storage）
    }
}

// collectAll：等待所有 Future 完成
intrusive_ptr<Future> collectAll(const List<intrusive_ptr<Future>>& srcs) {
    // 使用原子计数器和回调实现
}

// collectAny：返回第一个完成的 Future
intrusive_ptr<Future> collectAny(const List<intrusive_ptr<Future>>& srcs) {
    // 使用 atomic<bool> 确保只有一个 Future 被返回
}
```

## 类型推断（ivalue.cpp:86-155）

```cpp
TypePtr IValue::TagType<Type>::get(const IValue& v) {
    switch (v.tag) {
        case Tag::Tensor:
            return TensorType::create(v.toTensor());  // 推断 shape/dtype
        case Tag::GenericList:
            return ListType::create(v.toList().elementType());
        case Tag::GenericDict:
            return DictType::create(d.keyType(), d.valueType());
        case Tag::Object:
            return v.toObjectRef().type();  // ClassType
    }
}
```

## 辅助类型

### StrongTypePtr / WeakTypePtr（ivalue.h:1511-1585）

持有类型及其所属 CompilationUnit 的智能指针对：
- **StrongTypePtr**：拥有所有权（shared_ptr）
- **WeakTypePtr**：弱引用（避免 Object → CU → Graph → Object 循环引用）

---

## ROCm 相关内容
- 无直接 ROCm 代码（设备抽象通过 `c10::Device` 处理）

## Backward 相关内容
- `OptionalArray<T>`（lines 94-134）：用于 autograd 保存参数
- deepcopy 支持跨设备（用于梯度累积）
