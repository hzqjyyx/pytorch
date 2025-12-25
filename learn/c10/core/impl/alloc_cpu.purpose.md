## c10/core/impl/alloc_cpu.h

声明了CPU内存分配的公共接口：
- `alloc_cpu()` - 分配CPU内存
- `free_cpu()` - 释放CPU内存
- 可选的mimalloc包装函数（`c10_mi_malloc`, `c10_mi_calloc`等）

## c10/core/impl/alloc_cpu.cpp

实现CPU内存分配的核心逻辑：

**核心函数：`alloc_cpu()`**
1. 检查请求大小（不能为0或负数）
2. 根据平台选择内存分配方法：
   - Android: `memalign()`
   - Windows: `_aligned_malloc()` 或 `mi_malloc_aligned()`
   - Linux/Unix: `posix_memalign()`
3. 支持透明大页面(THP)优化：检测`THP_MEM_ALLOC_ENABLE`环境变量，对大缓冲区使用`madvise(MADV_HUGEPAGE)`
4. NUMA感知：将分配的内存移动到当前NUMA节点
5. 可选的内存填充：
   - 零填充（`caffe2_cpu_allocator_do_zero_fill`）
   - 垃圾填充（`caffe2_cpu_allocator_do_junk_fill`）- 填充`0x7fedbeef`模式用于调试

**辅助函数：`memset_junk()`**
- 用垃圾模式填充内存，该模式作为浮点数时为NaN，作为整数时为大值

**`free_cpu()`**
- 释放已分配的内存，平台相关实现

---

### 关键特性总结
- 跨平台内存分配（Android、Windows、Linux）
- 透明大页面支持
- NUMA优化
- 调试辅助（内存模式填充）
- 可选的mimalloc集成
