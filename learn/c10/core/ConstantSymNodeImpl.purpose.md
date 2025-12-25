## ConstantSymNodeImpl - 常量符号节点实现

**核心目的**：为 PyTorch 的符号形状推理系统提供常量值的包装类，支持在混合操作（常量与符号值）中进行适当的操作转发。

**主要设计**：

ConstantSymNodeImpl 是一个模板类，继承自 SymNodeImpl，用模板参数 T（int64_t 或 bool）来区分常量类型。内部使用 std::variant 存储实际的常量值。

**关键机制**：

1. **类型检查与提取**
   - is_int()、is_bool()、is_float()：检查常量类型
   - int_()、bool_()：安全地提取对应类型的值（含类型验证）
   - guard_int、guard_bool、guard_float：带文件行号记录的安全提取接口

2. **常量识别**
   - constant_int()、constant_bool()：返回 optional，允许类型不匹配时返回空值
   - is_constant() 返回 true、is_symbolic() 返回 false

3. **二元操作转发机制**（.cpp 中的宏定义）
   - 处理"常量 OP 嵌套符号值"的情况
   - 通过对调操作数和使用反向操作符（ROP）来转发：例如 `const <= symbol` 被转换为 `symbol >= const`
   - 支持的操作：eq, ne, ge, le, lt, gt, mul
   - 使用 intrusive_ptr 进行内存安全的转发

4. **字符串表示**
   - str()：int 类型返回数字字符串，bool 类型返回 "true" 或 "false"

**约束条件**：
- 仅支持 int64_t 和 bool 类型（不支持 float）
- 在运行时对类型操作进行严格检查（TORCH_CHECK、TORCH_INTERNAL_ASSERT）

**使用场景**：
当大负整数或布尔常量与符号值进行二元运算时，作为中间表示层确保正确的操作语义。

---

- 模板类包装常量（int64_t/bool）为符号节点
- 内部用 variant 存储值，constexpr 编译期类型判断
- 提供安全的类型检查与值提取接口
- 二元操作通过对调操作数转发给另一侧的符号节点
- 支持 eq/ne/ge/le/lt/gt/mul 运算，其余操作不支持
