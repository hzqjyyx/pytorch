## CPUAllocator 主要功能分析

### CPUAllocator.h (头文件)

定义了 CPU 内存分配器的公共接口和工具类：

- **ProfiledCPUMemoryReporter**: 内存分析报告类
  - `New()`: 记录新内存分配事件
  - `Delete()`: 记录内存释放事件
  - `OutOfMemory()`: 报告内存溢出事件
  - 内部使用哈希表追踪分配的内存块及其大小

- **公共 API 函数**:
  - `GetCPUAllocator()` / `SetCPUAllocator()`: 获取/设置 CPU 分配器
  - `GetDefaultCPUAllocator()`: 获取默认分配器
  - `GetDefaultMobileCPUAllocator()`: 获取移动设备默认分配器
  - `GetCPUCachingAllocator()` / `SetCPUCachingAllocator()`: 缓存分配器管理

### CPUAllocator.cpp (实现)

实现具体的内存分配逻辑：

**DefaultCPUAllocator** (非移动设备):
- `allocate()`: 分配内存，调用 `c10::alloc_cpu()`，记录到 ProfiledCPUMemoryReporter
- `ReportAndDelete()`: 释放时同时报告和删除
- `raw_deleter()`: 返回删除函数指针
- `copy_data()`: 数据复制

**DefaultMobileCPUAllocator** (移动设备版本):
- 模板类，支持前后卫字节（Pre/Post Guard Bytes）
  - Pre-Guard: 8 字节对齐（QNNPACK 需要）
  - Post-Guard: 16 字节（XNNPACK 需要）
- 智能分配策略：优先使用线程本地缓存分配器，其次用性能分析分配器，最后用默认分配
- `is_simple_data_ptr()`: 检查指针是否为简单数据指针

**ProfiledCPUMemoryReporter 实现**:
- `New()`: 维护分配表，更新总分配量，可选地报告到性能分析器和日志
- `Delete()`: 查表释放，处理未追踪的内存块（可能在分析启动前分配）
- `OutOfMemory()`: 记录分配失败事件

**全局变量与注册**:
- 编译时选择：`C10_MOBILE` 宏决定使用哪个分配器
- 使用 `REGISTER_ALLOCATOR` 宏注册为设备类型的默认分配器

### 核心流程

```
用户请求分配内存
    ↓
调用 allocate() → c10::alloc_cpu()
    ↓
记录到 ProfiledCPUMemoryReporter
    ↓
返回 DataPtr（包含指针、删除函数）
    ↓
（使用内存）
    ↓
调用删除函数 → ReportAndDelete()
    ↓
报告到分析器 → free_cpu()
```

---

### 关键特性总结

- **内存追踪**: 维护所有已分配内存块的大小映射
- **性能分析**: 支持向性能分析器报告内存事件
- **移动端优化**: 预分配卫字节防止 QNNPACK/XNNPACK 越界访问
- **灵活性**: 支持自定义分配器注册和优先级管理
- **缓存支持**: 提供可选的缓存分配器以减少碎片化
- **标志控制**: `caffe2_report_cpu_memory_usage` 标志控制详细日志输出
