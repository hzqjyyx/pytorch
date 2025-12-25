这个文件实现了双线性插值上采样 (Bilinear Upsampling) 的2D版本。

**主要功能分解：**

• **元函数定义** (`at::meta` 命名空间)
  - `upsample_bilinear2d`: 验证输入张量为4D，计算输出尺寸，设置输出张量的形状和内存格式
  - `_upsample_bilinear2d_aa`: 带抗锯齿(anti-aliasing)的版本，逻辑相同

• **CPU实现** (`at::native` 命名空间)
  - `upsample_bilinear2d_out_cpu`: 调用 `upsample_bilinear2d_kernel` 执行实际的双线性插值计算
  - `_upsample_bilinear2d_aa_out_cpu`: 调用 `_upsample_bilinear2d_aa_kernel` 执行带抗锯齿的插值

• **高级包装函数**
  - `upsample_bilinear2d`: 接收 `OptionalIntArrayRef` 或 `scale_factors`，计算最终输出尺寸和缩放因子后调用底层实现
  - `_upsample_bilinear2d_aa`: 同上，抗锯齿版本

• **分发宏** 
  - `DEFINE_DISPATCH`: 为四个核心函数注册分发机制，支持不同设备（CPU/GPU等）的多态实现
