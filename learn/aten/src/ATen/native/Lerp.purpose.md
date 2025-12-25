Lerp (Linear Interpolation) 实现了线性插值功能，用于计算两个张量之间的线性插值。

**核心功能：**

- **`lerp()` 模板函数** (Lerp.h:21-34)：执行实际的线性插值计算
  - 公式：当 weight < 0.5 时用 `self + weight * (end - self)` 计算，否则用 `end - (end - self) * (1 - weight)` 计算
  - 这两种方式数值稳定性不同，根据 weight 大小选择避免浮点精度问题
  - 支持标量和复数类型的 weight

- **`is_lerp_weight_small()` 辅助函数** (Lerp.h:11-18)：判断 weight 是否接近 0
  - 对于复数，避免调用 sqrt() 函数，直接比较实部平方 + 虚部平方

- **两种操作接口** (Lerp.cpp)：
  - `lerp_Tensor`：weight 是张量形式，支持按元素的不同 weight 值
  - `lerp_Scalar`：weight 是单个标量值

- **元函数 (Meta functions)** (Lerp.cpp:15-40)：处理张量形状和数据类型检查
  - 验证 self 和 end 数据类型一致
  - 使用 TensorIterator 配置广播规则和类型转换

- **实现函数 (Impl functions)** (Lerp.cpp:46-54)：调用对应的内核函数
  - 通过 `DEFINE_DISPATCH` 和设备类型分发到具体实现

**关键设计特点：**

- 采用 dispatch 模式支持不同设备（CPU/GPU）和数据类型
- 数值稳定性优化：根据 weight 值选择不同的计算方式
- 类型提升支持：weight 为 0-D 张量时自动类型转换
