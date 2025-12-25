# DynamicType 核心功能分析

## 设计目标

DynamicType 是为 TorchScript 设计的**轻量级类型系统**，主要解决移动端二进制体积膨胀问题。传统 JIT 类型系统同时用于编译和运行时，但移动端只需运行时功能，却引入了大量不必要的依赖（vtable、typeinfo、构造/析构函数等），导致二进制膨胀。

## 核心设计：基于位运算的类型表示

### 1. Tag 机制
每个类型用一个 32 位整数 `DynamicTypeBits` 表示，分为两部分：
- **控制位**（从 MSB 开始）：描述类型的特殊行为
  - `kDynamicCovariantTypeBit` (bit 31)：协变类型标记（如 Tuple）
  - `kDynamicAnyTypeBit` (bit 30)：Any 类型标记
- **数据位**（从 LSB 开始）：标识具体类型
  - 如 `Int = bit 3`, `Float = bit 4`, `List = bit 7`

### 2. 子类型判断优化
通过位运算实现高效的子类型检查（`dynamic_type.cpp:182-208`）：

```cpp
bool isSubtypeOfExt(const Type& rhs, ...) {
  auto other = create(rhs);
  
  // 1. 相同 tag 且相等
  if (tag_ == other->tag_ && equals(*other)) return true;
  
  // 2. 协变类型的参数协变检查
  if (contains(tag_, kDynamicCovariantTypeBit)) {
    if (compareArguments(*other, [](a, b) { return a.isSubtypeOf(b); }))
      return true;
  }
  
  // 3. 位包含关系：Number 包含 Int/Float/Complex
  if (contains(other->tag_, tag_)) return true;
  
  // 4. Optional 特殊处理
  if (other->tag_ == Tag::Optional) {
    if (isSubtypeOf(other->arguments_.elems[0].ty)) return true;
  }
}
```

### 3. 复合类型表示示例

**Number 类型**（`dynamic_type.h:33-35`）：
```cpp
Number = kDynamicIntTypeBit | kDynamicFloatTypeBit | kDynamicComplexTypeBit
```
这样 `Int` 自动成为 `Number` 的子类型（位包含关系）。

**Optional 类型**（`dynamic_type.h:41-43`）：
```cpp
Optional = DYNAMIC_TYPE_BIT(11) | kDynamicNoneTypeBit | kDynamicCovariantTypeBit
```
同时包含 None 位和协变位，支持 `Optional[T]` 的协变性。

## 主要组件

### 1. 类型构造（`dynamic_type.cpp:116-158`）

从 JIT Type 转换为 DynamicType：
- **ClassType**：特殊处理，存储在 union 的 `class_` 成员
- **普通类型**：通过 `FORALL_DYNAMIC_TYPES` 宏映射到对应 Tag
- **容器类型**（List/Dict/Tuple）：递归转换 `containedTypes()`
- **NamedTuple**：提取字段名存储在 `Arguments::label`

### 2. Arguments 结构（`dynamic_type.h:138-143`）

存储参数化类型的类型参数：
```cpp
struct Arguments {
  std::vector<LabeledDynamicType> elems;  // 支持带标签的参数
};

struct LabeledDynamicType {
  std::optional<std::string> label;  // 用于 NamedTuple
  DynamicTypePtr ty;
};
```

### 3. 类型恢复：fallback() 方法（`dynamic_type.cpp:236-317`）

服务端可将 DynamicType 转回完整 JIT Type：
- 基础类型：直接返回单例（`IntType::get()`）
- List/Dict/Optional：递归调用参数的 `fallback()`
- Tuple：处理 NamedTuple 的字段名恢复
- ClassType：返回存储的 `class_` 副本

### 4. IValue 类型提取（`dynamic_type.cpp:333-368`）

运行时从 IValue 推断类型：
```cpp
DynamicType::Ptr IValue::TagType<DynamicType>::get(const IValue& v) {
  switch (v.tag) {
    case Tag::Tensor: return DynamicTypeTrait<TensorType>::getBaseType();
    case Tag::GenericDict: 
      return DynamicTypeFactory::create<DictType>(d.keyType(), d.valueType());
    // ...
  }
}
```

## 内存管理特性

### Union 存储优化（`dynamic_type.h:209-212`）
```cpp
union {
  Arguments arguments_;   // 大多数类型
  ClassTypePtr class_;    // ClassType 特殊处理
};
```

### 手动生命周期管理（`dynamic_type.cpp:72-79`）
```cpp
~DynamicType() {
  if (tag_ == Tag::Class) {
    class_.~ClassTypePtr();
  } else {
    arguments_.~Arguments();
  }
}
```
需要显式调用析构，因为 union 不自动管理。

## 协变类型支持

Tuple 和 Optional 标记了协变位（`dynamic_type.h:38,42`），支持：
- `Tuple[Int, Float]` 是 `Tuple[Number, Number]` 的子类型
- `Optional[Int]` 是 `Optional[Number]` 的子类型

通过 `compareArguments` 递归检查每个参数的子类型关系（`dynamic_type.cpp:189-195`）。

## 伪类型映射（FAKE Types）

某些 JIT 类型映射到相同的 DynamicType Tag（`dynamic_type.h:63-67`）：
```cpp
ScalarType, Layout, SymInt, MemoryFormat → kDynamicIntTypeBit
```
运行时不区分这些类型，统一当作 Int 处理。`dynamicKind()` 方法故意忽略这些伪类型（`dynamic_type.cpp:226`）。

---

**其他特性：**
- 禁用拷贝/移动语义，强制使用 shared_ptr 管理
- 使用 `weak_from_this()` 检查对象是否由 shared_ptr 管理
- 字符串表示包含 Tag 数值和参数列表（`str()` 方法）
- 支持 VarType 的名称存储用于类型变量
