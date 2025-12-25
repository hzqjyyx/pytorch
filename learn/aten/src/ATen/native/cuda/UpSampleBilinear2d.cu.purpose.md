# UpSampleBilinear2d.cu 主要功能

这个文件实现了 2D 双线性上采样（bilinear upsampling）的 CUDA 前向和反向传播核函数。

## 核心实现逻辑

### 1. 前向传播 - 标准双线性插值

**核函数**: `upsample_bilinear2d_out_frame` (lines 35-80)

每个线程处理输出图像的一个像素位置 (h2, w2)：

1. **计算源坐标**（lines 54-66）：
   ```
   h1r = area_pixel_compute_source_index(rheight, h2, align_corners, false)
   w1r = area_pixel_compute_source_index(rwidth, w2, align_corners, false)
   ```
   得到浮点数坐标 h1r, w1r

2. **确定插值点**：
   - `h1 = floor(h1r)`, `w1 = floor(w1r)` - 左上角整数坐标
   - `h1p`, `w1p` - 是否有右下邻居（边界检查）
   - `h1lambda = h1r - h1`, `w1lambda = w1r - w1` - 插值权重
   - `h0lambda = 1 - h1lambda`, `w0lambda = 1 - w1lambda` - 补充权重

3. **双线性插值计算**（lines 70-76）：
   ```
   val = h0lambda * (w0lambda * idata[n][c][h1][w1] + 
                     w1lambda * idata[n][c][h1][w1+w1p]) +
         h1lambda * (w0lambda * idata[n][c][h1+h1p][w1] + 
                     w1lambda * idata[n][c][h1+h1p][w1+w1p])
   ```
   对 2x2 邻域的四个像素进行加权平均

### 2. NHWC 内存格式优化

**核函数**: `upsample_bilinear2d_nhwc_out_frame` (lines 84-128)

- 针对 channels-last (NHWC) 内存布局的优化版本
- 使用 `idx_cl()` 函数计算线性索引，适配 NHWC 排列
- 触发条件（line 290）：`memory_format == ChannelsLast && channels >= 16`

### 3. 抗锯齿上采样（Anti-aliased）

**核函数**: `upsample_gen2d_aa_out_frame` (lines 474-563)

使用更复杂的滤波器来减少上采样时的锯齿效应：

1. **计算支持域**（lines 496-502）：
   ```
   support_h = (height_scale >= 1.0) ? (filter_size * 0.5) * height_scale : filter_size * 0.5
   interp_height = ceil(support_h) * 2 + 1
   ```

2. **共享内存权重计算**（lines 520-543）：
   - 使用共享内存存储插值权重 `wx`, `wy`
   - 每个 block 内的线程协作计算权重
   - `threadIdx.y == 0` 计算 wx（所有 y 线程共享）
   - `threadIdx.x == 0` 计算 wy（所有 x 线程共享）

3. **两步插值**（lines 553-561）：
   - 先在 x 方向对每个 y 位置插值
   - 再在 y 方向对插值结果进行二次插值

4. **支持双线性和双三次**：
   - `BilinearFilterFunctor` - 双线性（line 880）
   - `BicubicFilterFunctor` - 双三次（line 908）

### 4. 关键优化策略

**内存格式选择**（lines 290-356）：
- ChannelsLast 路径：`channels >= 16` 时启用，更好的缓存局部性
- Contiguous 路径：通用实现，使用 `PackedTensorAccessor`

**共享内存管理**（lines 730-750）：
```cpp
weights_per_block = interp_width * block_x + interp_height * block_y + 
                    interp_height * block_y * block_x  // buffer
TORCH_CHECK(shmem_size <= sharedMemPerBlock)
```

**线程块配置**（lines 701-738）：
- block_x = warp_size（通常 32）
- block_y 根据共享内存容量动态计算
- 确保不超过设备限制

## 对外接口

- `upsample_bilinear2d_out_cuda` (line 847) - 标准双线性上采样
- `_upsample_bilinear2d_aa_out_cuda` (line 872) - 抗锯齿双线性上采样  
- `_upsample_bicubic2d_aa_out_cuda` (line 901) - 抗锯齿双三次上采样

---

### ROCm 和 Backward 相关
- Backward 函数使用 `fastAtomicAdd` 累积梯度到输入位置（lines 169-192, 235-258）
- 非确定性操作警告：由于原子操作，反向传播结果不可复现（lines 867, 894, 922）
- 支持 Half 和 BFloat16 精度（`AT_DISPATCH_FLOATING_TYPES_AND2`）
- ROCm 平台同样支持所有功能
