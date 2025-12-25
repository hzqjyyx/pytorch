• **文件目的**：为 CUDA 张量提供存储设置和形状重新调整的本地实现

• **set_cuda_()** - 初始化空 CUDA 张量
  - 创建新的 CUDA 存储（分配器来自 `getCUDADeviceAllocator()`）
  - 设置张量的存储为空（大小为 0）
  - 保持原有数据类型不变

• **set_storage_cuda_()** - 绑定存储到张量并调整形状
  - 验证存储设置的有效性（调用 `checkSetStorage`）
  - 设置存储偏移量
  - 调用 `resize_impl_cuda_()` 根据指定的尺寸和步幅调整张量形状
  - 支持可选的步幅参数（nullptr 时使用默认）

• **设计理由**：CUDA 实现与 CPU 实现分离，因为设备分配器获取方式不统一（避免不必要的调度开销）
