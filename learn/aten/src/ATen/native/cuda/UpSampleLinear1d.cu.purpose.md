## UpSampleLinear1d.cu 文件分析

**核心功能：**CUDA 实现 1D 线性插值上采样（upsample）

**主要流程：**

1. **upsample_linear1d_out_frame 内核** (lines 27-69)
   - 将输入张量上采样到指定的输出宽度
   - 对每个输出位置，计算对应的输入源位置
   - 使用线性插值混合相邻两个输入值

2. **关键计算步骤：**
   - 调用 `area_pixel_compute_source_index()` 计算输出像素在输入中的浮点坐标
   - 取整得到左邻近整数位置 w1
   - 计算权重：w0lambda（左权重）和 w1lambda（右权重）
   - 输出值 = w0lambda × input[w1] + w1lambda × input[w1+1]

3. **特殊情况处理：**
   - 若输入宽度 == 输出宽度，直接复制（无需插值）

4. **GPU 执行配置：**
   - 线程块大小：512
   - 网格大小：ceil(output_width / 512)

---

**Bullet Points:**

- CUDA 内核实现 1D 线性插值上采样
- 支持 align_corners 模式和动态缩放参数
- 使用线性插值混合相邻输入值生成输出
- 特殊情况优化：尺寸不变时直接复制
- 支持多种数据类型（浮点、半精度、BF16）
- GPU 并行化：每个线程处理一个输出宽度位置
- Backward 操作使用原子加法累积梯度
