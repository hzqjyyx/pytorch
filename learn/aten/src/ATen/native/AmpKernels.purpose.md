## AmpKernels 文件分析

这两个文件实现了 PyTorch 中自动混合精度 (Automatic Mixed Precision, AMP) 的核心操作。

**AmpKernels.h** 定义了两个函数指针类型和调度声明：

1. `_amp_foreach_non_finite_check_and_unscale_cpu__fn` - 接收缩放梯度列表、无穷值标志和反向缩放因子，检查梯度中的非有限值（NaN/Inf）并进行反缩放
2. `_amp_update_scale_cpu__fn` - 根据是否检测到无穷值，动态更新缩放因子（扩大或缩小），同时更新增长追踪器

**AmpKernels.cpp** 提供了这两个函数的 CPU 实现包装：

- `_amp_foreach_non_finite_check_and_unscale_cpu_()` - 委托给对应的 stub 实现，实际调用通过 `DEFINE_DISPATCH` 机制路由到设备特定的实现
- `_amp_update_scale_cpu_()` - 类似的委托模式，处理缩放因子的动态调整

这是一个**调度层** (dispatch layer)，使用 PyTorch 的 `DispatchStub` 机制将操作路由到不同设备后端（CPU、GPU等）的实现。

### 主要功能总结

- **梯度检查**: 检测梯度中的 NaN 或 Inf 值
- **动态缩放**: 根据梯度情况自动调整混合精度计算中的缩放因子
- **设备无关**: 通过 dispatch stub 支持多设备实现
- **批量操作**: 支持对多个梯度张量的并行处理
