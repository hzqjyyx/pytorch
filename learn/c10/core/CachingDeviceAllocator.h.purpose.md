# CachingDeviceAllocator.h 文件分析

## 主要功能

这个头文件定义了设备内存分配器的统计信息结构和工具函数。

### DeviceStats 结构体

记录内存分配器的统计数据，分为三类：

**计数类统计 (COUNT)：**
- `allocation` - 客户端代码请求的分配次数
- `segment` - 从设备内存分配的段数
- `active` - 活跃内存块数量（已分配或被流使用）
- `inactive_split` - 非活跃的分割内存块数量（无法通过设备内存释放操作释放）
- `oversize_allocations` - 从池分配的超大块数量
- `oversize_segments` - 需要malloc的超大块数量
- `num_alloc_retries` - 缓存刷新导致的malloc失败调用总数
- `num_ooms` - 缓存刷新后仍失败的OOM总数
- `num_sync_all_streams` - synchronize_and_free_events()调用次数
- `num_device_alloc` - 设备内存分配调用总数（包括mapped和malloced）
- `num_device_free` - 设备内存释放调用总数（包括unmapped和free）

**字节数统计 (SUM)：**
- `allocated_bytes` - 分配器分配的字节总数
- `reserved_bytes` - 分配器保留的字节总数（空闲+使用）
- `active_bytes` - 活跃内存块中的字节数
- `inactive_split_bytes` - 非活跃分割块中的字节数
- `requested_bytes` - 客户端代码请求的字节数

**配置参数：**
- `max_split_size` - 允许分割的最大块大小

### format_size() 函数

将字节数转换为可读格式（bytes/KiB/MiB/GiB），精度2位小数。

---

## 核心要点

- 定义了内存分配器性能监控的数据结构
- 通过多维度统计追踪内存使用和分配行为
- 支持诊断内存碎片化、OOM事件和分配性能
- 提供友好的内存大小格式化输出工具
