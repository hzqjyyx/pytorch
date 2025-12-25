这个文件实现了PyTorch中的数值积分操作，主要包含梯形法则（Trapezoid Rule）的相关函数。

**核心函数：**

- **`do_trapezoid(y, dx, dim)` 和 `do_trapezoid(y, dx, dim)`** - 梯形法则的核心实现，计算 $\sum_{i=1}^{n-1} dx_i \cdot (y_i + y_{i+1}) / 2$
  - 第一个版本接收Tensor型的dx（支持可变间距）
  - 第二个版本接收double型的dx（常数间距，公式简化）

- **`do_cumulative_trapezoid(y, dx, dim)`** - 累积梯形积分，返回沿指定维度的累积结果

- **`trapezoid(y, x, dim)` 和 `trapezoid(y, dx, dim)`** - 公开API
  - 处理输入验证和维度处理
  - 支持1维x或高维x的自动padding和reshape

- **`trapz(y, x, dim)` 和 `trapz(y, dx, dim)`** - 别名函数，直接调用trapezoid

- **`cumulative_trapezoid(y, x, dim)` 和 `cumulative_trapezoid(y, dx, dim)`** - 公开的累积积分API

- **`add_padding_to_shape(curr_shape, target_n_dim)`** - 工具函数，在shape前面补1以匹配目标维度数

- **`zeros_like_except(y, dim)`** - 创建与y形状相同但沿dim维度为1的零张量

**关键特性：**

- 支持任意维度的积分
- 支持可变间距（dx为Tensor）和常数间距（dx为标量）
- 自动处理维度匹配和广播
- 排除bool类型输入
- 使用符号形状（SymInt）支持动态维度
