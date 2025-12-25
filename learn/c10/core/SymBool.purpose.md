## SymBool 类的主要功能

SymBool 是 PyTorch 中用于表示符号布尔值的类，支持两种存储模式：

**存储机制：**
- 直接存储：简单的 `bool` 值存储在 `data_` 成员中
- 堆分配：复杂的符号表达式存储为 `SymNode` 指针 (`ptr_`)，通过 `is_heap_allocated()` 判断

**核心操作：**

1. **逻辑运算** (通过宏 DEFINE_BINARY 生成)
   - `sym_and()` / `operator&`：符号与操作
   - `sym_or()` / `operator|` / `operator||`：符号或操作
   - `sym_not()` / `operator~`：符号非操作
   - 针对常量值的优化：如果两个操作数都是常量，直接返回常量结果

2. **值提取与转换**
   - `maybe_as_bool()`：安全提取布尔值或符号常量值
   - `expect_bool()`：强制提取布尔值（若无法提取则触发检查失败）
   - `toSymNodeImpl()` / `toSymNodeImplUnowned()`：转换为符号节点

3. **符号节点包装**
   - `wrap_node()`：用基础节点包装当前值，若值为常量则直接包装常量

4. **断言与守卫**
   - `guard_bool()`：插入守卫强制符号值为其具体值并返回
   - `guard_size_oblivious()`：大小不敏感的守卫
   - `expect_true()`：预期值为真的断言
   - `has_hint()`：检查是否存在常量提示

5. **输出**
   - `operator<<`：输出运算符，常量显示数值，符号表达式显示其字符串表示

**宏定义辅助：**
- `TORCH_SYM_CHECK` / `TORCH_MAYBE_SYM_CHECK`：条件检查宏，适配 SymBool 和普通 bool

**设计特点：**

- 混合表示：优化常量路径，避免不必要的堆分配
- 延迟求值：符号表达式保留为节点树，延迟到实际需要时求值
- 类型安全：通过 `is_bool()` 检查确保 SymNode 确实表示布尔值

**主要用途：**

- 在 PyTorch 的符号形状推断系统中表示运行时布尔条件
- 支持动态形状计算中的条件逻辑（如循环条件、分支条件）
- 使用常量折叠优化避免符号表达式的不必要创建
