这个文件提供了**动态类型转换的工具函数**，用于在运行时根据 ScalarType 枚举值转换数据类型。

## 核心问题

文件注释详细说明了设计目的：如果不用动态转换，实现如 `add_` 这样支持类型提升的操作需要嵌套多个 `AT_DISPATCH_ALL_TYPES` 宏，会导致编译时间和二进制大小爆炸（N³ 种类型组合）。动态转换通过单个通用分支解决这个问题。

## 主要函数

**`fetch_and_cast<dest_t>(src_type, ptr)`**
- 从 void 指针读取数据，其类型由 src_type（ScalarType 枚举）指定
- 动态转换为静态类型 dest_t
- 通过 switch 语句对所有支持的 ScalarType 展开

**`cast_and_store<src_t>(dest_type, ptr, value)`**
- 将静态类型 src_t 的值转换为 dest_type 指定的动态类型
- 存储到 void 指针指向的内存

## 支持的类型

- AT_FORALL_SCALAR_TYPES_WITH_COMPLEX：所有基础数值类型和复数
- uint16_t, uint32_t, uint64_t：显式添加的无符号整数类型

## 特殊处理

**`DEFINE_UNCASTABLE` 宏**
- 为量化类型（QINT）提供特化版本
- 这些类型不能进行动态转换，只能直接加载/存储，使用 CUDA_KERNEL_ASSERT 验证类型匹配

---

## 要点总结

- **目的**：避免 N³ 类型组合爆炸，用单个动态分支替代嵌套宏展开
- **性能**：分支预测和 GPU warp 执行一致性保证性能损失可接受
- **实现**：宏展开 switch-case，覆盖所有 ScalarType 枚举值
- **限制**：某些类型（量化类型）无法转换，需要类型匹配断言
