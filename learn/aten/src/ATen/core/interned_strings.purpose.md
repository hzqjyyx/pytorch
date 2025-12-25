# interned_strings 核心功能

这是 PyTorch 的符号（Symbol）字符串化系统，用于高效管理和查询操作符、命名空间等字符串标识符。

## 主要组件

**Symbol 类型**
- 用整数 `unique_t` 表示字符串，避免重复存储
- 每个符号有命名空间前缀，格式为 `namespace::name`（如 `aten::add`）
- 内置符号在编译期定义为枚举值，运行时动态符号存储在映射表中

**InternedStrings 管理器**
- 维护双向映射：字符串 ↔ Symbol
  - `string_to_sym_`: 字符串到符号的查找表
  - `sym_to_info_`: 符号到元信息（命名空间、全限定名、非限定名）的数组
- 线程安全：使用 `mutex_` 保护动态符号的注册和查询

## 核心操作

**符号创建** (`_symbol`, `fromQualString`)
```cpp
Symbol sym = Symbol::fromQualString("aten::relu");
```
- 解析命名空间和名称
- 已存在则返回缓存的 Symbol
- 新符号则分配唯一 ID 并记录

**符号查询**
```cpp
const char* qual = sym.toQualString();    // "aten::relu"
const char* unqual = sym.toUnqualString(); // "relu"
Symbol ns = sym.ns();                      // namespaces::aten
```

**性能优化**
- 内置符号（通过 `FORALL_NS_SYMBOLS` 定义）使用编译期 switch-case 直接返回，无需加锁
- Mobile 构建（`C10_MOBILE`）移除 switch-case，统一走映射表查询以减少二进制体积

**命名空间系统**
```cpp
bool sym.is_aten();   // 检查是否属于 aten 命名空间
bool sym.is_prim();   // 检查是否属于 prim 命名空间
```

## 预定义符号类别

**interned_strings.h** 通过宏 `FORALL_NS_SYMBOLS` 定义数百个内置符号：
- **命名空间标识符**: `prim`, `aten`, `cuda`, `onnx`, `attr` 等
- **控制流**: `prim::If`, `prim::Loop`, `prim::Return`
- **张量操作**: `prim::TupleConstruct`, `aten::append`, `aten::dim`
- **类型转换**: `aten::Int`, `aten::Float`, `aten::Bool`
- **ONNX 导出**: `onnx::Add`, `onnx::Conv`, `onnx::MatMul`
- **属性键**: `attr::name`, `attr::inplace`, `attr::scope`

## 域名系统
```cpp
sym.domainString();  // "org.pytorch.aten"
Symbol::fromDomainAndUnqualString("org.pytorch.aten", "relu");
```
用于 ONNX 等外部格式的标准化表示。

---

**忽略的相关内容：**
• ROCm 相关功能
• Backward 自动微分相关符号和机制
