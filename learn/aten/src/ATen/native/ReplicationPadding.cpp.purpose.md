# ReplicationPadding.cpp 主要功能

这个文件实现了 PyTorch 中的 **Replication Padding** 操作，用于对张量进行边界复制填充。

## 核心操作

1. **元函数定义** (at::meta 命名空间)
   - `TORCH_META_FUNC(replication_pad1d)` - 1D 填充的形状推导
   - `TORCH_META_FUNC(replication_pad2d)` - 2D 填充的形状推导
   - `TORCH_META_FUNC(replication_pad3d)` - 3D 填充的形状推导
   - 这些函数计算输出张量的大小并验证输入有效性

2. **CPU 实现函数** (at::native 命名空间)
   - `replication_pad1d_out_cpu` - 1D 填充实现
   - `replication_pad2d_out_cpu` - 2D 填充实现
   - `replication_pad3d_out_cpu` - 3D 填充实现
   - 这些函数调用具体的 kernel 执行实际计算

3. **前向传播逻辑**
   - 接收输入张量和 paddingSize 参数
   - paddingSize 定义每个维度的左/右(上/下/前/后)填充大小
   - 通过边界元素复制扩展张量维度

## 关键细节

- **维度处理**：支持带或不带批次维度的张量 (2D/3D for 1D padding, 3D/4D for 2D, 4D/5D for 3D)
- **内存格式**：2D 和 3D 填充调用 `suggest_memory_format()` 优化内存布局
- **校验**：检查 paddingSize 大小、输出尺寸有效性、gradOutput 维度匹配

---

- Replication padding 通过复制边界像素值来填充张量
- 支持 1D、2D、3D 三种维度的填充操作
- 包含元函数(形状推导)和实现函数(CPU 计算)
- 后向传播通过累积梯度回传到原始张量位置
- 使用 DEFINE_DISPATCH 宏支持多设备(CPU/CUDA/ROCm)
