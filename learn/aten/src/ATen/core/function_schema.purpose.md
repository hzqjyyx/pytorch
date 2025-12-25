# FunctionSchema 核心功能分析

这两个文件实现了 PyTorch 算子的函数签名（schema）系统，用于描述和验证算子的接口定义。

## 核心数据结构

### Argument（参数）
表示函数的单个参数或返回值：
- **类型信息**：`type_` (表面类型) 和 `real_type_` (实际类型，如 ScalarType vs int)
- **元数据**：参数名、默认值、是否仅限关键字参数 (`kwarg_only_`)
- **别名信息**：`alias_info_` 用于别名分析，标识参数是否可写
- **特殊属性**：`N_` 表示固定长度列表的大小，`is_out_` 标记输出参数

### FunctionSchema（函数签名）
描述完整的算子接口：
- **标识**：`name_` 包含算子名和重载名 (`OperatorName`)
- **参数列表**：`arguments_` 输入参数，`returns_` 返回值
- **可变参数**：`is_vararg_` 和 `is_varret_` 支持不定长参数/返回值
- **别名分析**：`alias_kind_` 控制别名分析策略

## 主要功能模块

### 1. 类型克隆与转换

**`cloneWithRealTypes(bool with_symint)`** (function_schema.cpp:21-54)
- 将 fake type 转换为 real type，处理 SymInt 相关类型
- 输入参数根据 `with_symint` 选择性保留 SymInt 类型
- 返回值始终使用 real type

**`cloneWithRemappedTypes()`** (function_schema.cpp:554-571)
- 通过自定义函数映射所有参数和返回值的类型
- 用于类型系统转换

### 2. 别名分析

**核心方法：**

**`mapTypeToAliasTypeSet(const TypePtr& type)`** (function_schema.cpp:98-145)
- 将类型映射到别名类型集合，相同别名集合的类型可能互相别名
- 处理容器类型递归展开：`ListType`, `DictType`, `TupleType`, `UnionType`, `OptionalType`
- 标量类型返回 `nullopt`（不可变）

**`may_alias(lhs, rhs)`** (function_schema.cpp:147-175)
- 检查两个参数是否可能直接别名
- 条件：类型集合可别名 + 别名信息的 afterSets 有交集

**`may_contain_alias(lhs, rhs, bidirectional)`** (function_schema.cpp:177-199)
- 检查容器嵌套的别名关系
- 处理 wildcard 别名（`*`）与容器内元素的别名
- `bidirectional=false` 仅检查 lhs 是否包含 rhs 的别名

**`getAliasTypeSetContainedTypes()`** (function_schema.cpp:70-96)
- 递归提取容器内所有嵌套类型，用栈遍历避免重复

### 3. 兼容性检查

**向后兼容 (Backward Compatibility)：** `isBackwardCompatibleWith()` (function_schema.cpp:376-436)

新 schema 可替代旧 schema 的条件：
- 名称、重载名、vararg/varret 标志、返回值数量一致
- 返回值类型协变（新类型是旧类型的子类型）
- 参数类型逆变（旧类型是新类型的子类型）
- 新增参数必须有默认值且位于 out 参数之前
- Out 参数位置对齐验证

**向前兼容 (Forward Compatibility)：** `isForwardCompatibleWith()` (function_schema.cpp:438-516)

额外限制：
- Out 参数数量必须完全相同
- 新增参数的默认值不能是容器类型（避免序列化问题）
- 所有新参数必须在旧参数和 out 参数之间

### 4. 序列化与格式化

**`operator<<(ostream, FunctionSchema)`** (function_schema.cpp:201-288)
- 生成可被 SchemaParser 解析的字符串格式
- 特殊处理：
  - 单返回值加括号情况：`(str,t)[]` 避免解析歧义
  - Kwarg-only 参数前插入 `*,`
  - Vararg/varret 用 `...` 表示

**`operator<<(ostream, Argument)`** (function_schema.h:550-624)
- 格式：`Type(alias)? name=default_value`
- 处理 sized list：`int[3]`
- 默认值序列化：字符串加引号，统一默认值的列表简化为单值

### 5. 运行时验证

**`formatTypeMismatchMsg()`** (function_schema.cpp:353-374)
- 生成类型不匹配的详细错误信息
- 包含参数位置、期望类型、实际类型、函数签名

**`findErrorInKwargs()`** (function_schema.cpp:518-551)
- 检测未知关键字参数
- 检测位置参数与关键字参数重复指定

**`checkAndNormalizeInputs()`** (在 function_schema_inl.h 中实现)
- 验证输入参数类型匹配
- 填充缺失的默认值

## 辅助功能

**`findFirstOutArg()`** (function_schema.cpp:290-298)
- 查找第一个 out 参数的索引，用于分离常规参数和输出参数

**`getCorrectList(SchemaArgType)`** (function_schema.cpp:13-19)
- 根据类型返回 arguments 或 returns 列表

**`canAliasTypeSetsAlias()`** (function_schema.cpp:56-68)
- 判断两个别名类型集合是否有交集

## 设计关键点

1. **Fake Type vs Real Type**：支持类型推断过程中的临时类型表示，最终转换为真实类型
2. **SymInt 特殊处理**：符号整数用于动态形状推断，在某些上下文保持符号形式
3. **别名分析分层**：直接别名 → 容器包含别名 → wildcard 别名，逐级检查
4. **兼容性约束**：新参数必须有默认值 + 位置约束，保证代码演化的平滑过渡
5. **Out 参数分离**：特殊处理输出参数的验证和序列化

---

**简要提及的相关内容：**
- ROCm 支持：Hipblaslt/Rocblas 相关的 GEMM 调优实现
- Backward compatibility：旧模型加载新代码的兼容性保证机制
