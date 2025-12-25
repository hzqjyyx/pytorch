## RangeFactories.cu 主要功能

这个文件实现了 PyTorch CUDA 上的数值范围生成操作，包含以下核心部分：

**GPU 内核执行框架** (RangeFactories.cu:38-66)
- `elementwise_kernel_with_index`: 通用 CUDA 内核，按索引执行函数并写入输出张量
- `gpu_kernel_with_index`: 主机端包装函数，配置线程块和网格，启动内核执行

**四个范围生成函数：**

1. **linspace_cuda_out** (RangeFactories.cu:72-121)
   - 生成指定步数的均匀间隔数值序列
   - 从 start 到 end，共 steps 个点
   - 分别处理整数类型和浮点/复数类型

2. **logspace_cuda_out** (RangeFactories.cu:123-175)
   - 生成对数间隔数值序列
   - 计算 base^start 到 base^end，共 steps 个点
   - 同样区分整数和浮点/复数类型

3. **range_cuda_out** (RangeFactories.cu:177-211)
   - 生成从 start 到 end（不含），步长为 step 的序列
   - 使用累加类型提高精度

4. **arange_cuda_out** (RangeFactories.cu:213-272)
   - 类似 range，但有额外的精度处理
   - 专门针对 int64_t 类型的特殊逻辑，避免精度丧失
   - 包含 size 计算的验证和警告机制

**关键特性：**

- **非连续张量处理**：临时创建连续张量后再复制回原张量
- **类型分派**：使用 AT_DISPATCH 宏支持多种标量类型
- **精度优化**：采用中点向外计算（halfway split）减少浮点误差累积
- **尺寸计算**：使用累加类型保持计算精度，检查溢出和无穷值
