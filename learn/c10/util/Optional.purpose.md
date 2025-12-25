## Optional.h

这个文件是 PyTorch 的 c10 库中对 `std::optional` 的包装和扩展。主要功能：

1. **标准库重导出** (第13-24行)
   - 条件编译下，将 `std::optional`、`std::nullopt`、`std::bad_optional_access` 等标准库组件暴露到 `c10` 命名空间
   - 仅在非 FBCODE_CAFFE2 和非 C10_NODEPRECATED 条件下启用

2. **deprecated 函数：value_or_else** (第36-55行)
   - 两个重载版本，接收 const 引用和右值引用的 optional
   - 功能：如果 optional 有值则返回该值，否则调用传入的函数获取默认值
   - 带 static_assert 检查函数返回类型必须可转换为 optional 的值类型
   - 标记为已废弃，建议使用 `std::optional::value_or` 替代

## Optional.cpp

仅包含头文件包含，无实现代码。

---

**总结：**

- 条件编译地重导出 `std::optional` 相关类型到 `c10` 命名空间
- 提供已废弃的 `value_or_else` 函数（传入 callable 计算默认值）
- 主要用于向后兼容性和 API 一致性
