# ATen/core/jit_type.h 功能分析

这是PyTorch JIT类型系统的核心头文件，定义了TorchScript运行时的完整类型层次结构。

## 核心功能

### 1. 类型层次结构

**基础类型体系**：
- `AnyType`: 类型层次顶端，所有类型的超类型
- `NumberType`: 数值类型基类
  - `IntType`: Python int
  - `FloatType`: Python float  
  - `ComplexType`: Python complex
- `BoolType`, `StringType`, `NoneType`: 基本标量类型

**符号化类型**（用于动态形状追踪）：
- `SymIntType`, `SymFloatType`, `SymBoolType`: 支持在维度值上追踪算术运算

### 2. 容器类型

**单元素容器** (`SingleElementType` 模板)：
- `ListType`: `List[T]`，对应Python列表
- `OptionalType`: `Optional[T]`，等价于 `Union[T, None]`
- `FutureType`, `AwaitType`, `RRefType`: 异步/分布式类型

**复合容器**：
- `DictType`: `Dict[K, V]`，键类型受限（int/float/complex/Tensor/device/string）
- `TupleType`: 支持命名元组和匿名元组
- `UnionType`: 联合类型，`OptionalType`是其特殊形式

### 3. Tensor类型系统

`TensorType` 是最复杂的类型，包含：

**静态属性**：
- `scalar_type_`: 数据类型（可选）
- `device_`: 设备信息（可选）
- `requires_grad_`: 梯度需求（可选）
- `undefined_`: 是否为undefined tensor

**形状表示**：
- `SymbolicShape sizes_`: 符号化形状，支持：
  - 完全未知的rank
  - 已知rank但未知维度
  - 部分已知维度（如 `[2, ?, 3]`）
  - 完全静态形状

**步幅属性** (`VaryingShape<Stride> strides_`)：
```cpp
struct Stride {
  std::optional<size_t> stride_index_;  // 步幅排序索引
  std::optional<bool> contiguous_;      // 连续性标记
  std::optional<size_t> stride_;        // 具体步幅值
};
```

支持融合内核优化：即使维度被转置，只要a、b、c以相同方式转置且连续，就能简化索引逻辑。

**核心方法**：
- `withSizes()`, `withStrides()`: 附加形状/步幅信息
- `withSymbolicShapes()`: 支持符号化形状
- `dimensionedOnly()`: 保留rank但丢弃具体尺寸
- `contiguous()`: 生成连续版本
- `merge()`: 合并两个TensorType的公共信息
- `isComplete()`: 判断类型信息是否完整（除autograd外全部已知）

### 4. 用户自定义类型

**类与接口**：
- `InterfaceType`: 抽象方法列表，支持鸭子类型
  - 子类型规则：方法是超集 + module接口需module实现
- `ClassType`（引用但未在此文件定义）
- `FunctionType`: 封装 `torch::jit::Function*`

**枚举相关**：
- `AnyEnumType`: 所有枚举的超类型
- `EnumerationType<K>`: 枚举类型模板
- 特殊枚举：`ScalarTypeType`, `MemoryFormatType`, `LayoutType`（映射为int但保留语义）

### 5. 类型操作工具

**类型合并与统一**：
```cpp
std::optional<TypePtr> unifyTypes(
  const TypePtr& t1, 
  const TypePtr& t2,
  bool default_to_union = false,  // 找不到超类型时是否返回Union
  const TypePtr& type_hint = nullptr
);
```
- 两个TensorType返回动态版本
- 不支持NumberType作为{Float, Int, Complex}的超类型（缺少算子支持）

**类型变量匹配**（用于泛型函数）：
```cpp
MatchTypeReturn matchTypeVariables(
  const TypePtr& formal,   // 形式类型（含类型变量）
  const TypePtr& actual,   // 实际类型
  TypeEnv& type_env        // 类型环境映射
);
```

**形状合并**：
- `SymbolicShape::merge()`: 只保留静态且相等的维度
- `VaryingShape<T>::merge()`: 使用 `merge_primitive` 合并每个元素

### 6. C++类型到JIT类型映射

`getTypePtr<T>()` 通过特化实现：
```cpp
getTypePtr<at::Tensor>()          → TensorType::get()
getTypePtr<std::vector<T>>()      → ListType::get("vector", inner)
getTypePtr<std::optional<T>>()    → OptionalType::get(inner)
getTypePtr<std::tuple<Ts...>>()   → TupleType::create({Ts...})
getTypePtr<c10::Dict<K,V>>()      → DictType::get("Dict", K, V)
```

**Fake类型支持**（`getFakeTypePtr<T>()`）：
- `SymInt` → `IntType` (fake模式)
- `SymFloat` → `FloatType` (fake模式)
- 用于不支持符号化形状的场景

### 7. 类型安全检查

```cpp
void checkNoAny(
  const Type& base,
  const char* what,
  const std::string& attrname,
  const TypePtr& attrtype
);
```
防止`AnyType`出现在命名类型（class/namedtuple/interface）中，否则Pickler会丢失动态类型信息。

### 8. 辅助功能

**类型字符串化**：
- `str()`: 用户友好表示（如 `"Tensor"`, `"int[]"`）
- `annotation_str()`: Python类型注解格式（如 `"List[int]"`）
- `repr_str()`: 带额外信息（如 `"Tensor (inferred)"`）

**类型判断**：
- `isComplete()`: VaryingShape/SymbolicShape/TensorType完整性检查
- `hasFreeVariables()`: 是否包含未绑定的类型变量
- `containsAnyType()`: 递归检查是否包含AnyType

**通用超类型**：
- `AnyListType`: 所有List的超类型
- `AnyTupleType`: 所有Tuple的超类型
- `AnyClassType`: 所有ClassType的超类型

## 关键设计模式

1. **全局单例**：基础类型通过 `SingletonTypePtr<T>` 管理，避免重复实例
2. **SharedType**：容器/复合类型使用 `shared_ptr` 管理生命周期
3. **CRTP模式**：`SingleElementType<K, T>` 提供统一接口
4. **渐进式类型细化**：从 `TensorType::get()` → `withDim()` → `withSizes()` → `contiguous()`
5. **类型环境传递**：通过 `TypeEnv` (unordered_map) 绑定类型变量

## 典型使用场景

- **形状推断**：通过 `SymbolicShape` 追踪动态维度，支持编译优化
- **类型检查**：验证操作输入类型匹配
- **序列化**：Pickler依赖静态类型信息重建类型标签
- **JIT编译**：根据类型信息生成特化代码（如连续内存优化）
- **自动求导**：`requires_grad`字段参与梯度图构建

---

**简要列出（ROCm/Backward相关）**：
• ROCm相关：无直接内容，底层 `CUDADevice.h`, `CUDASparse.h` 等包含但此文件纯类型定义
• Backward相关：`requires_grad_` 字段支持autograd，`undefined_` 标记零梯度张量（`UndefinedTensorImpl`）
