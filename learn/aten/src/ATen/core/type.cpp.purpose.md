## 文件主要功能

这是 PyTorch 类型系统的核心实现文件，负责运行时类型表示、类型推断和类型关系判断。

### 核心职责

**1. 类型的字符串表示 (operator<<)**
- 实现了类型的可读输出格式
- TensorType: 输出形如 `Float32Tensor(3, 224, 224)` 或带 strides/device 信息
- 容器类型: `List[Int]`, `Optional[Tensor]`, `Tuple[Int, Float]`
- 支持通过环境变量 `PYTORCH_JIT_TYPE_VERBOSITY` 控制详细程度

**2. 单例类型的懒初始化**
- 使用静态局部变量实现线程安全的单例模式
- 包括基础类型: IntType, FloatType, BoolType, StringType 等
- 容器类型工厂: ListType::ofInts(), ListType::ofFloats() 等
- 避免静态初始化顺序问题

**3. 泛型容器类型缓存 (get 方法)**
- OptionalType::get(inner): 缓存 `Optional[T]` 类型
- ListType::get(identifier, inner): 缓存 `List[T]` 类型
- DictType::get(identifier, key, value): 缓存 `Dict[K, V]` 类型
- 使用 flat_hash_map + mutex 实现线程安全缓存

**4. 类型统一 (unifyTypes)**
- 找到两个类型的最小公共父类型
- 处理子类型关系: 如果 t1 是 t2 的子类型，返回 t2
- TensorType 的 merge: 统一 shape/dtype/device 信息
- None 与非 Optional 统一为 Optional
- 递归处理容器类型: Tuple, Optional, Future
- 回退到 unshaped 类型再尝试统一

**5. 类型变量匹配 (matchTypeVariables)**
- 模式匹配中的类型变量绑定
- 将形式参数中的类型变量 (VarType) 绑定到实际类型
- 递归处理容器: List[t] 匹配 List[Int] 时绑定 t=Int
- 支持 Tuple 到 List 的松弛匹配 (如果元素类型可统一)
- 验证类型变量的一致性绑定

**6. 类型变量求值 (tryEvalTypeVariables)**
- 将类型中的类型变量替换为具体类型
- 例如 `List[List[t]]` 在 t=Int 时变为 `List[List[Int]]`
- 递归处理嵌套的容器类型

**7. 子类型关系判断**

*Type::isSubtypeOfExt 基础实现*:
- AnyType 是所有类型的父类型
- Optional[T] 可以接受 T 的子类型
- UnionType 需要是其中某个分支的子类型

*NoneType::isSubtypeOfExt*:
- None 是任何 Optional 类型的子类型

*NumberType::isSubtypeOfExt*:
- NumberType 特殊处理为 Union[Int, Float, Complex]

*TupleType::isSubtypeOfExt*:
- 协变规则: 每个元素类型必须是对应位置的子类型
- 无名元组不是具名元组的子类型
- 具名元组可以是无名元组的子类型
- 具名元组间需要字段名完全匹配

*ListType::isSubtypeOfExt*:
- 任何 ListType 都是 AnyListType 的子类型

*InterfaceType::isSubtypeOfExt*:
- 检查所有方法签名的兼容性
- 方法需要满足逆变/协变规则
- 非 module interface 不能是 module interface 的子类型

**8. TupleType 的具名元组支持**

*createNamed 工厂方法*:
- 创建带字段名和默认值的具名元组
- 生成对应的 FunctionSchema
- 检查默认值不能是 Tensor (避免别名问题)

*names() 方法*:
- 返回字段名列表

*equals 方法*:
- 除了类型相等，schema 也必须完全匹配

**9. 辅助函数**

- `typeKindToString()`: TypeKind 枚举转字符串
- `elementTypeCanBeInferredFromMembers()`: 判断容器元素类型能否从成员推断
- `containsAnyType()`: 递归检查类型树是否包含 AnyType
- `checkNoAny()`: 确保模块/类/具名元组不包含 Any 类型

### 代码组织特点

- 使用 static_assert 验证 shared_ptr 和 SingletonOrSharedTypePtr 的内存布局
- 大量使用 std::optional 表示可选信息
- 类型缓存使用 ska::flat_hash_map 优化性能
- 通过 castRaw/cast/expect 进行类型转换和检查

### ROCm/Backward 相关
- 无直接相关内容
