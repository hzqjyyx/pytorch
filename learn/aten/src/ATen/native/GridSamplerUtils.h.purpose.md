**GridSamplerUtils.h 主要功能：**

- **定义枚举类型**：GridSamplerInterpolation（双线性、最近邻、双三次插值）和 GridSamplerPadding（零填充、边界、反射）

- **check_grid_sampler_common**：验证输入和网格张量的基本属性
  - 检查张量是否已定义
  - 验证设备和布局一致性
  - 确保批处理大小相同
  - 校验网格最后一维大小等于输入空间维度数

- **check_grid_sampler_2d**：验证 2D grid sampler 的张量维度
  - 确保输入和网格都是 4D（批次、通道、高、宽）

- **check_grid_sampler_3d**：验证 3D grid sampler 的张量维度和插值模式
  - 确保输入和网格都是 5D
  - 禁止 3D 输入使用双三次插值（仅支持 4D）

- **cond_cudnn_grid_sampler**：判断是否可使用 cuDNN 加速
  - 验证输入/网格可被 cuDNN 接受
  - 检查是否支持 32 位索引寻址
  - 限制输入为 4D 且通道数 ≤ 1024
