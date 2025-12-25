# GridSampler 核心功能分析

## 主要用途

Grid Sampler 实现了基于网格的空间变换采样，用于神经网络中的可微分图像/体积采样。主要应用场景包括 Spatial Transformer Networks (STN)、图像变形、几何变换等。

## 核心机制

### 坐标变换流程

1. **归一化坐标到像素坐标** (`grid_sampler_unnormalize` - GridSampler.h:26-35)
   - 输入：[-1, 1] 归一化坐标
   - 输出：像素索引值
   - `align_corners=true`: -1→0, +1→(size-1)，缩放因子 (size-1)/2
   - `align_corners=false`: -1→-0.5, +1→(size-0.5)，缩放因子 size/2

2. **边界处理** (`compute_coordinates` - GridSampler.h:142-159)
   - **Zeros padding**: 边界外为 0（默认行为）
   - **Border padding**: 裁剪到图像边界 (`clip_coordinates`)
   - **Reflection padding**: 按边界反射坐标 (`reflect_coordinates`)

3. **插值采样** (GridSampler.cpp:610-703)
   - **Bilinear**: 双线性插值，加权 4 个邻近像素
   - **Nearest**: 最近邻插值，取最近像素值
   - **Bicubic**: 双三次插值（仅 2D），加权 4x4=16 个邻近像素

## 实现细节

### 2D Grid Sampler (GridSampler.cpp:918-955)

**输入形状**:
- `input`: [N, C, H_in, W_in] - 输入图像批次
- `grid`: [N, H_out, W_out, 2] - 采样网格，最后维度为 (x, y) 坐标

**处理流程** (以 `_grid_sampler_2d_cpu_fallback` 为例，GridSampler.cpp:596-707):

```
对每个输出像素 (n, h, w):
  1. 从 grid[n, h, w] 读取归一化坐标 (x, y)
  2. 转换为像素坐标 (ix, iy)
  3. 根据插值模式采样:
     - Bilinear: 计算 4 角权重 (nw, ne, sw, se)，加权求和
     - Nearest: 四舍五入取最近像素
     - Bicubic: 4x4 网格三次插值
  4. 对所有通道 C 应用相同采样
```

**关键优化**:
- 检查 gather 指令溢出 (GridSampler.cpp:938-942)：AVX gather 使用 32 位有符号偏移，大张量可能溢出，降级到 fallback 实现
- 量化支持 (GridSampler.cpp:926-928)：QUInt8 类型使用专门的量化实现 `_grid_sampler_2d_cpu_quantized`
- 向量化：主路径调用 `grid_sampler_2d_cpu_kernel` (cpu/GridSamplerKernel.h)，利用 SIMD 加速

### 3D Grid Sampler (GridSampler.cpp:41-201)

**输入形状**:
- `input`: [N, C, D_in, H_in, W_in] - 输入体积批次
- `grid`: [N, D_out, H_out, W_out, 3] - 采样网格，最后维度为 (x, y, z) 坐标

**Bilinear 插值** (实为 trilinear，GridSampler.cpp:99-178):
- 8 个角点命名：tnw/tne/tsw/tse (顶层) + bnw/bne/bsw/bse (底层)
- 权重计算基于对角顶点距离（例如 `tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)`）
- 对每个通道累加 8 个有效邻居的加权值

### 量化实现 (GridSampler.cpp:446-555)

**特点**:
- 仅支持 Bilinear 插值 (GridSampler.cpp:461-463)
- 在量化域直接计算，无需反量化
- 边界外使用 `zero_point` 而非 0 (GridSampler.cpp:538-547)
- 输出保持量化格式 (QUInt8)

### 辅助函数

**坐标裁剪** (`clip_coordinates` - GridSampler.h:57-59)
```cpp
return std::min(clip_limit - 1, std::max(in, 0))  // 限制在 [0, clip_limit-1]
```

**坐标反射** (`reflect_coordinates` - GridSampler.h:88-104)
```cpp
in = |in - min|                    // 距边界距离
extra = in % span                  // 一个周期内余数
flips = floor(in / span)           // 反射次数
return flips % 2 == 0 ? extra + min : span - extra + min
```

**边界检查** (GridSampler.h:204-210)
- `within_bounds_2d`: h∈[0,H) && w∈[0,W)
- `within_bounds_3d`: d∈[0,D) && h∈[0,H) && w∈[0,W)

**安全累加** (`safe_add_2d/3d` - GridSampler.h:237-253)
- 仅在坐标有效时累加梯度，避免越界访问

## 并行化策略

所有实现使用 `at::parallel_for(0, N, 0, [&](...){...})` 按批次 N 并行：
- 每个线程处理批次子集 [start, end)
- 内层循环串行遍历空间维度和通道
- 避免通道维度的竞争写入

---

## 简要列举

**Backward 相关**:
- `grid_sampler_3d_backward_cpu_impl` (GridSampler.cpp:204-442)：计算 grad_input 和 grad_grid
- `_grid_sampler_2d_cpu_fallback_backward` (GridSampler.cpp:711-916)：2D 梯度计算
- `*_set_grad` 函数系列：同时返回值和导数（链式法则）
- `get_cubic_coefficients_grad` (GridSampler.h:279-296)：三次卷积核导数

**ROCm 相关**:
- 无明显 ROCm 特定代码，ROCm 后端复用 CPU 实现或有独立 CUDA/HIP 内核
