## Dimname 模块功能分析

**核心目的**：管理张量维度的命名系统，支持命名维度（named tensors）功能。

**主要类和结构**：

`Dimname` 结构体：
- 表示单个维度的名称
- 包含两种类型：`BASIC`（具名维度）和 `WILDCARD`（通配符维度，用 `*` 表示）
- 内部存储 `Symbol` 对象作为名称标识和 `NameType` 枚举表示类型

**核心功能**：

1. **名称创建与验证**
   - `fromSymbol()`: 从 Symbol 创建 Dimname，自动验证名称合法性
   - `isValidName()`: 检查字符串是否为有效标识符（仅允许字母、下划线、数字，首字符不能是数字）
   - `wildcard()`: 创建通配符维度实例

2. **维度匹配与统一**
   - `unify()`: 尝试统一两个维度名称，返回 `std::optional<Dimname>`
     - 若其中一个是通配符，返回另一个
     - 若两个相同，返回该维度
     - 否则返回 `std::nullopt`（不兼容）
   - `matches()`: 简化版 unify，仅返回是否匹配的布尔值

3. **输出与比较**
   - 重载 `operator<<`: 将 Dimname 输出为字符串（WILDCARD 输出为 "None"，其他输出为 `'name'` 格式）
   - 重载 `operator==` 和 `operator!=`: 基于 Symbol 相等性比较

**使用场景**：
- PyTorch 命名张量（Named Tensors）特性的基础设施
- 维度间类型检查与匹配验证

---

**功能总结**：

- 定义维度名称的数据结构和表示方法
- 验证名称的有效性（Python 标识符规则）
- 实现维度名称的统一/匹配算法
- 提供序列化和比较操作
