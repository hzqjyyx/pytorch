这个文件实现了 CUDA 上的 2D 平均池化操作。主要包含以下内容:

## 核心 CUDA Kernel

**1. avg_pool2d_out_cuda_frame (NCHW 布局)**
- 处理 Contiguous 内存格式 (batch, channels, height, width)
- 每个线程计算输出的一个元素
- 通过 `(n * channels + c) * height * width` 索引输入数据
- 对池化窗口内的所有元素求和后除以 divide_factor

**2. avg_pool2d_out_cuda_frame_nhwc (NHWC 布局)**
- 处理 ChannelsLast 内存格式 (batch, height, width, channels)
- 索引方式改为 `(h * width + w) * channels`，channel 维度在最内层
- 其余逻辑与 NCHW 版本相同

## 关键计算逻辑

**池化窗口边界计算:**
```
hstart = ph * stride_h - pad_h
hend = min(hstart + kernel_h, height + pad_h)
pool_size = (hend - hstart) * (wend - wstart)  // 包含 padding 的窗口大小
hstart = max(hstart, 0)  // 裁剪到有效输入范围
hend = min(hend, height)
```

**除数选择 (divide_factor):**
- 如果指定了 `divisor_override`，使用该值
- 如果 `count_include_pad=true`，使用完整窗口大小 `pool_size`
- 否则只计算有效区域大小 `(hend - hstart) * (wend - wstart)`

**累加与平均:**
```cuda
accscalar_t aveval = 0;
for (int h = hstart; h < hend; ++h) {
  for (int w = wstart; w < wend; ++w) {
    aveval += bottom_slice[h * width + w];
  }
}
top_data[index] = aveval / divide_factor;
```

## 前向传播入口函数 (avg_pool2d_out_cuda)

**主要流程:**
1. 参数转换：将 int64_t 参数安全降级为 int
2. 计算输出尺寸：`pooling_output_shape` 考虑 ceil_mode
3. 内存布局检测：根据 `input.suggest_memory_format()` 选择 Contiguous 或 ChannelsLast
4. Kernel 启动配置：
   - `num_threads = min(maxThreadsPerBlock, 1024)`
   - `num_blocks = ceil_div(count, num_threads)`
5. 类型分发：`AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16)` 支持 FP32/FP64/FP16/BF16
6. 根据内存格式调用对应的 NHWC 或 NCHW kernel

**特殊处理:**
- 空输出时跳过 kernel 启动 (`if (count != 0)`)
- ChannelsLast 格式需要 `empty_tensor_restride` 调整步幅
- 使用 `accscalar_t = acc_type<scalar_t, true>` 提升累加精度

---

**其他内容:**
- **Backward kernels** (`avg_pool2d_backward_out_cuda_frame` / `_nhwc`): 计算梯度反向传播，每个输入元素累加所有相关池化窗口的梯度
- **ROCm 特定优化**: `gesm10x` 检测 (SM10.x 架构) 限制 double 类型线程数为 768
- **索引类型分发**: 大张量使用 int64_t 索引 (`AT_DISPATCH_INDEX_TYPES`)
