UpSampleBicubic2d.cu 实现了 CUDA 上的双三次插值上采样（bicubic upsampling）。

## 核心功能

**前向传播 (upsample_bicubic2d_out_frame)**
- 将小分辨率图像放大到指定尺寸
- 使用双三次插值算法在 x、y 两个方向分别进行插值
- 对每个输出像素，先在 x 方向对 4 个源像素做三次插值，再在 y 方向对这 4 个结果做三次插值
- 支持 `align_corners` 模式控制源目标像素对齐方式

**关键实现细节**
- 每个 CUDA 线程处理一个输出像素位置
- 使用 `area_pixel_compute_source_index` 计算输出像素对应的源图像浮点坐标
- 通过 `cubic_interp1d` 和 `upsample_get_value_bounded` 进行三次插值和边界处理
- 支持 Float、Half、BFloat16 等多种数据类型

**接口层**
- `TORCH_IMPL_FUNC(upsample_bicubic2d_out_cuda)` 实现前向运算
- 调用 `upsample_bicubic2d_out_cuda_template` 完成具体计算

---

**总结**
- 双三次插值上采样的 CUDA 实现
- 输出像素通过 4×4 邻域源像素的三次插值得到
- 支持可配置的对齐和缩放参数
