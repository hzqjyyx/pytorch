# UpSample 模块主要功能

## 核心功能

**UpSample.cpp/h** 实现了 PyTorch 中的上采样（upsampling）功能，用于调整张量的空间维度大小。主要提供了计算输出尺寸的工具函数和各种插值算法的辅助函数。

## UpSample.cpp (compute_output_size 函数)

`compute_output_size` 是唯一的实现函数，负责根据输入参数计算上采样后的输出尺寸：

- **输入参数**：
  - `input_size`: 完整输入张量尺寸
  - `output_size`: 可选的目标输出尺寸
  - `scale_factors`: 可选的缩放因子

- **逻辑**：
  1. 计算空间维度数 = 输入尺寸 - 2（去掉 batch 和 channel 维度）
  2. 如果提供 `output_size`，直接返回
  3. 如果提供 `scale_factors`，遍历空间维度，用 `input_size[i+2] * scale_factor[i]` 计算输出维度
  4. 两者必须提供且只能提供一个，否则报错

## UpSample.h 核心内容

### 1. 插值模式的 scale 计算策略（Note [compute_scales_value] 和 [area_pixel_compute_scale]）

文档说明了两种 `scale_factor` 行为：
- **recompute_scale_factor = True**（当前默认）：用 scale_factor 计算 output_size，再用 input_size 和 output_size 反推新的 scale 用于插值
- **recompute_scale_factor = False**：直接使用用户提供的 scale_factor

**对齐策略**：
- `align_corners=True`: 保留角点中心，scale = (input_size - 1) / (output_size - 1)
- `align_corners=False`: 整个范围缩放，scale = input_size / output_size

### 2. 尺寸检查函数

- `upsample_1d_common_check`: 检查 1D 上采样的输入/输出尺寸（输入应为 [N, C, W]）
- `upsample_2d_common_check`: 检查 2D 上采样（输入应为 [N, C, H, W]）
- `upsample_3d_common_check`: 检查 3D 上采样（输入应为 [N, C, D, H, W]）
- `upsample_2d_shape_check`: 更详细的 2D 形状检查，支持梯度输出验证

### 3. 插值计算核心函数

**Scale 计算**：
- `compute_scales_value<scalar_t>`: 计算缩放比例，优先使用提供的 scale，否则用 input_size/output_size
- `area_pixel_compute_scale<scalar_t>`: 区域像素缩放计算，考虑 align_corners 参数

**源索引计算**：
- `area_pixel_compute_source_index<scalar_t>`: 计算源张量索引位置
  - align_corners: `src_idx = scale * dst_index`
  - 否则: `src_idx = scale * (dst_index + 0.5) - 0.5`

**最近邻插值索引**：
- `nearest_neighbor_compute_source_index`: OpenCV INTER_NEAREST 匹配的实现（有已知 bug，为了向后兼容保留）
- `nearest_neighbor_exact_compute_source_index`: 精确实现，公式为 `round((output_index + 0.5) * scale - 0.5)`
- `nearest_idx`: 旧版实现，对 scale=1 和 scale=2 做了特殊优化
- `nearest_exact_idx`: 新版精确实现

**双三次插值**：
- `cubic_convolution1/2`: 双三次卷积权重计算
- `get_cubic_upsample_coefficients`: 计算 4 个双三次插值系数（A=-0.75）
- `cubic_interp1d`: 1D 双三次插值

**线性插值辅助**：
- `compute_source_index_and_lambda`: 计算两个相邻源索引和插值权重 λ
- `guard_index_and_lambda`: 防止浮点精度问题导致索引越界

**边界处理**：
- `upsample_get_value_bounded`: 读取边界内的值（裁剪到有效范围）
- `upsample_increment_value_bounded`: 累加值到边界内位置

### 4. Dispatch 声明

使用 `DECLARE_DISPATCH` 宏声明了各种上采样核心函数的调度器，支持：
- **最近邻**: nearest1d/2d/3d (普通版和 exact 版)
- **线性插值**: linear1d, bilinear2d, trilinear3d
- **双三次插值**: bicubic2d
- **抗锯齿版本**: bilinear2d_aa, bicubic2d_aa

这些调度器会根据设备类型（CPU/CUDA）选择对应的实现。

### 5. 类型定义

- `scale_t = std::optional<double>`: 缩放因子类型
- 各种函数指针类型（如 `upsampling_nearest1d`, `upsampling_bilinear2d` 等）定义了不同插值方法的签名
- `nearest_idx_fn_t`: 最近邻索引函数的函数指针类型

---

**简要说明（ROCm & Backward）**：
- 文件中未直接包含 ROCm 特定代码，ROCm 支持通过 dispatch 机制实现
- Backward 相关：
  - 声明了所有前向操作对应的 backward kernel dispatch
  - `apply_grad_input` 函数模板处理 BFloat16/Half 类型的梯度累加，使用 float 作为累加缓冲区提高精度
