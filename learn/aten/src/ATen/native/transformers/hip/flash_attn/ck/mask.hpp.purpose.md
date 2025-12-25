这个文件定义了Flash Attention中的掩码(mask)机制，主要用于在注意力计算中选择性地屏蔽某些位置。

**核心组件：**

- **mask_enum枚举**：定义4种掩码类型
  - `no_mask`：无掩码
  - `mask_top_left`：从左上角的因果掩码(causal mask)
  - `mask_bottom_right`：从右下角的因果掩码
  - `window_generic`：通用窗口掩码

- **mask_info结构体**：存储掩码配置信息
  - `type`：掩码类型
  - `y, x`：掩码的高度和宽度坐标
  - `left, right`：滑动窗口注意力(Sliding Window Attention)的左右边界大小

- **serialize()方法**：将掩码配置序列化为字符串表示
  - `n`：无掩码
  - `t(left:right)`：上-左掩码，带窗口大小
  - `b(left:right)`：下-右掩码，带窗口大小
  - `g(y:x)`：通用掩码，带坐标

- **decode()静态方法**：从字符串解析掩码配置
  - 支持`xformer`风格的滑动窗口注意力格式(`xt:size`、`xb:size`)
  - 支持精确窗口指定格式(`t:left,right`、`b:left,right`、`g:y,x`)
  - 支持简化因果掩码格式(`t`、`b`或数字枚举值)
  - 自动计算掩码坐标并配置左右窗口大小

**功能总结：**

- 封装注意力掩码的类型、参数和序列化/反序列化逻辑
- 支持因果掩码和滑动窗口掩码两大类
- 提供字符串<->掩码配置的双向转换
- 与`ck_tile`库集成进行掩码坐标计算
