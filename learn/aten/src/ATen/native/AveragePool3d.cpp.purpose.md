这个文件实现了3D平均池化操作的CPU版本，主要包含前向传播和反向传播两部分。

## 核心功能

### 1. Meta函数：参数验证和输出形状计算

**avg_pool3d (lines 20-86)**
- 解析和验证kernel_size、stride、padding参数（支持单个int或3个int的tuple）
- 检查输入张量维度（4D或5D，即是否包含batch维度）
- 计算输出尺寸：使用`pooling_output_shape`函数计算时间、高度、宽度维度的输出大小
- 调用`pool3d_shape_check`验证所有维度的合法性
- 设置输出张量形状和选项

### 2. CPU前向传播实现

**avg_pool3d_out_frame模板函数 (lines 156-244)**
核心计算逻辑：
- 使用`at::parallel_for`对slices维度并行化
- 三层嵌套循环遍历输出的每个位置(ti, i, j)
- 对每个输出位置：
  1. 计算池化窗口范围：`tstart = ti * dT - padT`（时间、高度、宽度三个维度）
  2. 处理边界：将范围限制在有效输入区域内
  3. 计算除数因子：根据`divisor_override`、`count_include_pad`决定用哪个值
  4. 在池化窗口内求和：三层循环累加输入值
  5. 除以除数因子得到平均值

**avg_pool3d_out_cpu (lines 248-331)**
- 处理参数解析（与meta函数相同的kernel、stride、padding提取逻辑）
- 将输入转为连续内存布局
- 分两种情况：
  - **非batch模式**（4D输入）：直接调用`avg_pool3d_out_frame`
  - **batch模式**（5D输入）：使用`at::parallel_for`对batch维度并行，每个batch调用`avg_pool3d_out_frame`
- 使用`AT_DISPATCH_FLOATING_TYPES_AND`支持多种数据类型（float、double、long）

### 3. 关键参数说明

- `count_include_pad`: 是否在计算平均值时包含padding区域（影响除数计算）
- `divisor_override`: 可选的自定义除数，覆盖默认行为
- `ceil_mode`: 使用ceiling模式计算输出形状（而非floor）

**其他内容：**
- Backward相关：`avg_pool3d_backward` meta函数（lines 88-147）、`avg_pool3d_backward_out_frame`模板（lines 336-416）、`avg_pool3d_backward_out_cpu`实现（lines 420-508）用于反向传播梯度计算
- ROCm相关：无（此文件纯CPU实现）
