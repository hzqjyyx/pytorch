这个文件定义了两个内联验证函数，用于检查 im2col 和 col2im 操作的输入张量形状和参数的有效性。

**col2im_shape_check** 函数验证：
- kernel、stride、dilation 必须大于0
- padding 必须非负
- 输入张量必须是2D或3D（支持batch模式），且非batch维度非零
- 输入第1维大小必须能被 kernel_height × kernel_width 整除
- 输入第2维长度必须等于计算出的sliding blocks数量（n_blocks_height × n_blocks_width）
- 计算出的sliding blocks高宽都必须至少为1
- 输出空间尺寸必须为正

**im2col_shape_check** 函数验证：
- kernel、stride、dilation 必须大于0
- padding 必须非负
- 输入张量必须是3D或4D（支持batch模式），且除batch外的维度都非零
- 根据输入空间尺寸、kernel、stride、padding、dilation计算输出height和width
- 计算出的输出高宽都必须至少为1

**核心职责：**
- 防止无效参数导致的运算错误
- 提供详细的错误消息便于调试
- 使用 TORCH_CHECK 宏进行验证
