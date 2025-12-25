这两个文件是 PyTorch 的 CUDA 模块中用于测试 CUDA 库是否正常工作的简单测试工具。

**CUDATest.h** 是头文件，声明了一个公共导出的函数：
- `c10_cuda_test()` - 返回当前 CUDA 设备的索引

**CUDATest.cpp** 是实现文件，包含三个函数：

1. **has_cuda_gpu()** - 检查系统是否有可用的 CUDA GPU
   - 调用 `cudaGetDeviceCount()` 获取 GPU 数量
   - 返回 `true` 如果数量不为 0

2. **c10_cuda_test()** - 主要测试函数
   - 如果系统有 CUDA GPU，则调用 `cudaGetDevice()` 获取当前设备索引
   - 返回设备索引（没有 GPU 时返回 0）

3. **c10_cuda_private_test()** - 私有测试函数
   - 简单返回 2，仅供内部使用

**主要功能总结：**

- 检测系统是否有可用的 CUDA GPU
- 获取当前活跃的 CUDA 设备索引
- 验证 CUDA 运行时库是否正常工作
- 使用宏 `C10_CUDA_CHECK` 和 `C10_CUDA_IGNORE_ERROR` 处理 CUDA API 调用的错误
