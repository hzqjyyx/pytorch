这个文件是 PyTorch 中用于测试**已弃用的 lambda 式内核注册 API** 的测试文件。它验证了通过 `c10::RegisterOperators()` 和 lambda 函数注册算子内核的各种场景是否正常工作。

## 核心测试内容

### 1. 基本注册和调用
- 测试通过 `.op()` 方法注册 lambda 内核后能正确调用
- 测试通过构造函数直接注册
- 测试在同一个 registrar 中注册多个算子
- 测试跨多个 registrar 注册
- 测试 registrar 析构后算子schema自动注销

### 2. 不同返回类型
- **无返回值**: `void` 返回或 `std::tuple<>` 空元组
- **单一返回值**: int、Tensor、Tensor列表、int列表
- **多返回值**: 返回 `std::tuple` 包含 Tensor、int、列表、可选值、Dict 等

### 3. 不同参数类型
**值传递 vs 引用传递**:
- Tensor 的 `const Tensor&` 引用传递
- Tensor 的值传递
- 测试两种方式在有/无返回值时都能正常工作

**基础类型**:
- int64_t 标量
- std::string 字符串
- std::vector<int64_t> 整数列表
- std::vector<Tensor> Tensor列表

**复杂嵌套类型**:
- `Dict<string, Tensor>` 字典
- `std::unordered_map<string, string>` 无序映射
- `Dict<str, int[]>` 字典套列表
- `Dict<str, Dict<int,str>[]>` 字典套列表套字典
- `Dict<str, int[])[]` 列表套字典套列表

### 4. 可选参数处理
- 测试 `Tensor?`、`int?`、`str?` 等可选类型
- 验证传入 `c10::IValue()` (None) 时的处理
- 测试可选参数在单返回值和多返回值场景下的行为

### 5. Fallback 内核
- 测试没有 Tensor 参数的算子（只能作为 fallback 内核）
- 这种算子无法通过 Tensor 确定 dispatch key，需要 fallback 机制

### 6. Schema 推导
- 测试不显式指定 schema，框架自动从 lambda 签名推导
- 验证推导出的 schema 与预期一致

### 7. Boxed vs Unboxed 调用
- 测试通过 `callOp()` 的 boxed 调用（参数打包成 IValue）
- 测试通过 `callOpUnboxed<>()` 的 unboxed 调用（直接类型调用）

### 8. 错误检测
测试注册时参数不匹配会抛出异常:
- 参数数量不匹配（schema 声明 2 个参数，lambda 只接受 1 个）
- 参数类型不匹配（schema 声明 float，lambda 接受 int64_t）
- 返回值数量不匹配（schema 声明返回 int，lambda 返回 void）
- 返回值类型不匹配（schema 声明返回 Tensor，lambda 返回 int）

## 测试辅助工具
- `dummyTensor(DispatchKey)`: 创建指定 dispatch key 的测试 Tensor
- `extractDispatchKey(Tensor)`: 提取 Tensor 的 dispatch key
- `callOp()` / `callOpUnboxed<>()`: 调用算子
- `expectDoesntFindOperator()`: 验证算子不存在

## 关键技术点
1. **类型擦除**: lambda 签名通过模板推导转换为统一的 boxing 接口
2. **Schema 验证**: 运行时检查 schema 字符串与 lambda 实际签名是否匹配
3. **IValue 转换**: 测试各种 C++ 类型与 IValue 之间的双向转换
4. **生命周期管理**: RegisterOperators 的 RAII 机制，析构时自动注销

---

**忽略内容简述**:
- ROCm 相关: 文件中未涉及 ROCm 特定内容
- Backward 相关: 测试中使用 `at::AutoDispatchBelowAutograd` 跳过自动微分，但不涉及反向传播逻辑测试
