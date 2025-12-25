## 文件功能分析

这两个文件实现了从 C++ 函数类型自动推导 `FunctionSchema` 的机制，是 PyTorch 操作注册系统的核心部分。

### 核心组件

**ArgumentDef 结构体** (infer_schema.h:19-25)
- 编译期轻量级参数定义，包含两个函数指针
- `getTypeFn`: 获取实际的 c10 类型
- `getFakeTypeFn`: 获取伪类型（用于模式匹配或验证）
- 目的是减小二进制体积，避免在模板中直接构造 `Argument` 对象

**类型检查** (infer_schema.h:34-44)
- `checkStaticTypes()` 在编译期验证 C++ 类型的合法性
- 限制整数类型只能是 `int8_t`、`int64_t`、`bool`
- 禁止 `float` 类型（必须用 `double`）

**参数向量创建** (infer_schema.h:47-56)
- `createArgumentVectorFromTypes()` 利用 index_sequence 在编译期生成参数数组
- 使用可变模板参数展开每个参数的类型信息

**返回值处理** (infer_schema.h:76-106)
- 支持三种返回值形式：
  - `std::tuple<A, B, C>` → 多返回值
  - 单类型 `A` → 自动包装为单元组
  - `void` → 空返回值

**运行期函数** (infer_schema.cpp:25-41)
- `make_function_schema()` 在运行时从编译期数据构建 `FunctionSchema` 对象
- `createArgumentVector()` 将 `ArgumentDef` 数组转换为 `Argument` 向量，赋予自动名称 `_0, _1, _2...`

**Schema 验证** (infer_schema.cpp:44-90)
- `findSchemaDifferences()` 对比两个 FunctionSchema
- 逐项检查参数/返回值数量和类型
- 先比较指针相等性（快速路径），再做深层次类型比较
- 返回差异描述或 `std::nullopt`（无差异）

### 工作流程

1. 编译期：模板推导函数签名，创建 `ArgumentDef` 数组和类型检查
2. 运行期：调用 `make_function_schema()` 生成实际的 `FunctionSchema` 对象
3. 验证：`findSchemaDifferences()` 用于比对推导的 Schema 与用户指定的 Schema

---

- 实现 C++ 函数类型到 FunctionSchema 的自动推导
- 编译期生成参数/返回值定义，减小二进制体积
- 运行期统一构造 Argument 对象
- 支持元组解包、void 返回、多返回值
- 编译期类型安全检查（禁用不支持的类型）
- 运行期 Schema 差异检测和诊断
