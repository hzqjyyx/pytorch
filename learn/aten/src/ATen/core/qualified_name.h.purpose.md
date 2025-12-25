## QualifiedName 结构体

这个文件定义了一个用于处理点分名称（如 "foo.bar.baz"）的 `QualifiedName` 结构体。

### 核心设计

- **原子存储**：将点分名称分解为原子（atoms），存储在 `std::vector<std::string>` 中
- **缓存优化**：预计算并缓存三个派生值，避免重复计算
- **隐式转换**：支持从字符串隐式构造

### 构造方式

1. **从字符串**：`QualifiedName("foo.bar.baz")` - 自动按 '.' 分割
2. **从原子向量**：`QualifiedName({"foo", "bar", "baz"})`
3. **从前缀+名称**：`QualifiedName(prefix, "baz")` - 追加单个原子

### 主要接口

- `qualifiedName()`：完整的点分名称 "foo.bar.baz"
- `prefix()`：前导部分 "foo.bar"
- `name()`：最后一个原子 "baz"
- `atoms()`：原子向量
- `isPrefixOf(other)`：检查是否为另一个名称的前缀
- 相等性比较操作符

### 验证机制

- 不允许空原子
- 不允许原子内包含分隔符 '.'
- 使用 `TORCH_CHECK` 和 `TORCH_INTERNAL_ASSERT` 验证

### Hash 支持

- 为 `std::unordered_map` 等容器提供 hash 实现
- 基于 `qualifiedName_` 的哈希值

---

**主要用途**：表示分层的限定名称（如 PyTorch 中的模块路径或命名空间），提供高效的访问和比较操作。
