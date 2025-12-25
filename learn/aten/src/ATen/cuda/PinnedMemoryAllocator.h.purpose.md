这个文件的主要功能：

* 提供获取 CUDA 固定内存分配器的接口
* 通过 `getPinnedMemoryAllocator()` 函数返回一个 `at::Allocator*` 指针
* 该函数实际上委托给 `getCachingHostAllocator()`，使用缓存主机内存分配器来管理固定内存
* 固定内存（Pinned Memory）是 GPU 可直接访问的主机内存，用于提高 GPU-CPU 数据传输效率
* 使用缓存策略而不是每次都分配/释放，以减少内存分配开销
* 在 `at::cuda` 命名空间中定义，供 ATen CUDA 模块使用
