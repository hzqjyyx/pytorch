StorageUtils 提供了三个核心功能：

**1. new_shm_fd_storage()**
- 创建基于文件描述符的共享内存存储实现
- 使用 MapAllocator 分配共享内存区域
- 设置标志：EXCLUSIVE（独占）、KEEPFD（保留文件描述符）、UNLINK（解链接）
- 返回 c10::StorageImpl 的 intrusive_ptr

**2. storage_copy()**
- 将源存储数据复制到目标存储
- 通过创建临时张量包装存储对象，利用张量的 copy_ 操作执行实际复制
- 支持异步非阻塞复制（non_blocking 参数）
- 适配不同设备（CPU/GPU）的存储

**3. share_memory_()**
- 原地将张量存储转换为共享内存存储
- 仅作用于 CPU 张量且未被共享的情况
- 工作流程：
  - 检查张量是否已为共享内存（通过 MapAllocator 检查）
  - 创建新的共享内存存储
  - 复制原有数据到新存储
  - 替换原张量的 data_ptr 和 allocator

---

• 共享内存存储创建与管理
• 存储间数据复制（支持异步操作）
• 张量存储共享内存化转换
• 多设备兼容（CPU/GPU）
• 镜像 PyTorch Python 接口的 share_memory_() 行为
