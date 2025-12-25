- **文件位置**: `aten/src/ATen/cuda/CUDAUtils.h`
- **命名空间**: `at::cuda`
- **核心功能**: 提供 CUDA 设备检查工具函数

**主要内容**：

- `check_device(ArrayRef<Tensor> ts)`: 检查张量列表中的所有张量是否都在当前 CUDA 设备上
  - 空列表返回 `true`
  - 获取当前设备信息（`current_device()`）
  - 遍历每个张量，验证其设备与当前设备匹配
  - 任何张量设备不匹配则返回 `false`

**依赖关系**：
- 依赖 `<ATen/cuda/CUDAContext.h>` 获取 CUDA 上下文信息
