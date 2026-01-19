- **ceil_div(a, b)**：计算 `a / b` 的向上取整，使用公式 `(a + b - 1) / b`
  - 仅支持整数类型（通过 `std::is_integral_v<T>` 约束）
  - 标记为 `C10_ALWAYS_INLINE` 和 `C10_HOST_DEVICE`，可在 CPU 和 GPU 上执行

- **round_up(a, b)**：将 `a` 向上舍入到 `b` 的最近倍数
  - 实现为 `ceil_div(a, b) * b`
  - 例如：`round_up(10, 3)` = `ceil_div(10, 3) * 3` = `4 * 3` = `12`

- **用途**：内存对齐、缓冲区大小计算等需要向上取整的场景
