## PixelShuffle 核心功能分析

### PixelShuffle.h - 形状检查
定义了两个内联验证函数：

**check_pixel_shuffle_shapes**
- 验证输入至少有3维 (batch, channel, height, width)
- 检查upscale_factor为正数
- 检查通道数能被upscale_factor²整除

**check_pixel_unshuffle_shapes**
- 验证输入至少有3维
- 检查downscale_factor为正数
- 检查height和width都能被downscale_factor整除

### PixelShuffle.cpp - 核心实现

**pixel_shuffle_cpu** (PixelShuffle.cpp:23-45)
- 输入: (B, C, H, W)，upscale_factor
- 输出: (B, C/(r²), H*r, W*r)，其中r=upscale_factor
- 流程：计算输出形状 → 创建输出张量 → 调用kernel处理数据

**pixel_unshuffle_cpu** (PixelShuffle.cpp:47-73)
- 输入: (B, C, H, W)，downscale_factor
- 输出: (B, C*d², H/d, W/d)，其中d=downscale_factor
- 流程：计算输出形状 → 创建输出张量 → 调用kernel处理数据

**math_pixel_shuffle** (PixelShuffle.cpp:75-114)
- 纯数学实现（不依赖特定backend）
- 三步变换：
  1. Reshape: C → (oc, r, r, H, W)
  2. Permute: 重排维度 (oc, H, r, W, r)
  3. Reshape: 合并维度 → (oc, H*r, W*r)

**math_pixel_unshuffle** (PixelShuffle.cpp:116-155)
- 纯数学实现
- 三步变换：
  1. Reshape: (C, H, W) → (C, oh, d, ow, d)
  2. Permute: 重排维度 → (C, d, d, oh, ow)
  3. Reshape: 合并维度 → (C*d², oh, ow)

### 关键特征

- **内存优化**: 使用suggest_memory_format()保持内存格式一致性
- **空张量处理**: 处理numel()==0的边界情况
- **连续性**: contiguous()确保数据布局连续
- **调度**: DEFINE_DISPATCH注册kernel实现供不同backend使用

---

### 总结

- **pixel_shuffle**: 上采样操作，将多通道→少通道+更大空间
- **pixel_unshuffle**: 下采样操作，将少通道→多通道+更小空间
- **CPU版本**: 委托给kernel实现具体计算
- **Math版本**: 通过reshape+permute+reshape完成变换，不依赖特定优化kernel
