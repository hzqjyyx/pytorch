## PeerToPeerAccess 模块功能说明

这个模块管理 CUDA 多 GPU 环境中的点对点（P2P）访问能力。

**核心数据结构：**
- `p2pAccessEnabled_`：一个 int8_t 向量，记录 GPU 对之间的 P2P 访问状态
  - `1` 表示允许复制
  - `0` 表示禁止复制
  - `-1` 表示未知（需要查询）
- `num_devices_`：缓存的 GPU 设备总数

**主要函数：**

1. **`init_p2p_access_cache(int64_t num_devices)`** (PeerToPeerAccess.cpp:19-32)
   - 初始化 P2P 访问缓存
   - 创建 num_devices × num_devices 的二维缓存矩阵
   - 将同一设备的对角线元素设为 1（自己总是可以访问自己）

2. **`get_p2p_access(source_dev, dest_dev)`** (PeerToPeerAccess.cpp:36-59)
   - 查询两个 GPU 设备之间是否支持 P2P 访问
   - 逻辑流程：
     1. 检查缓存是否已有结果，有则直接返回
     2. 缓存未知时，调用 `cudaDeviceCanAccessPeer()` 查询硬件能力
     3. 如果支持 P2P，调用 `CUDACachingAllocator::enablePeerAccess()` 启用功能
     4. 结果保存到缓存中

**关键特点：**
- 使用缓存避免重复的硬件查询
- 自动启用支持的 P2P 访问
- 输入验证确保设备索引有效

**主要功能点：**

- P2P 访问状态缓存管理
- GPU 设备间数据直接转移能力检测
- 硬件 P2P 能力的延迟初始化和动态启用
- 支持 NVIDIA CUDA 设备的多 GPU 通信优化
