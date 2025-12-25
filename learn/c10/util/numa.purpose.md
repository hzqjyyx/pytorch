# NUMA 功能模块分析

NUMA（Non-Uniform Memory Access）是多处理器系统中的内存架构。这个模块提供了对 NUMA 节点的管理和内存绑定功能。

## 核心机制

**条件编译**：代码根据平台支持情况分为两部分：
- Linux + C10_USE_NUMA 宏定义 → 启用完整 NUMA 功能
- 其他情况 → 返回默认值/空操作

**全局标志**：`FLAGS_caffe2_cpu_numa_enabled` 控制 NUMA 功能的启用

## 主要函数功能

### IsNUMAEnabled() c10/util/numa.h:14
检查 NUMA 是否可用，需要同时满足：
- 标志位开启
- 系统支持 NUMA（`numa_available() >= 0`）

### NUMABind(int numa_node_id) c10/util/numa.h:19
将当前线程绑定到指定 NUMA 节点：
- 分配 NUMA 位掩码
- 设置对应节点位
- 调用 `numa_bind()` 执行绑定

### GetNUMANode(const void* ptr) c10/util/numa.h:24
查询指针所在的 NUMA 节点：
- 通过 `get_mempolicy()` 系统调用获取内存策略
- 返回节点 ID 或 -1（不可用）

### GetNumNUMANodes() c10/util/numa.h:29
获取系统 NUMA 节点总数：
- 调用 `numa_num_configured_nodes()`

### NUMAMove(void* ptr, size_t size, int numa_node_id) c10/util/numa.h:34
将内存迁移到指定 NUMA 节点：
- 计算页面对齐的起始地址（`getpagesize() - 1`）
- 构造位掩码标识目标节点
- 通过 `mbind()` 系统调用执行迁移，使用 `MPOL_BIND | MPOL_MF_MOVE | MPOL_MF_STRICT` 标志

### GetCurrentNUMANode() c10/util/numa.h:39
获取当前 CPU 所在 NUMA 节点：
- 通过 `sched_getcpu()` 获取 CPU ID
- 调用 `numa_node_of_cpu()` 查询对应节点

## 关键设计细节

- **安全初始化**：注释指出分配可能在静态初始化期间触发，故不使用 VLOG
- **内存对齐**：NUMAMove 中的位掩码操作确保节点 ID < 64
- **错误处理**：使用 `TORCH_CHECK` 和 `AT_ASSERT` 进行验证
- **非负数检查**：负 numa_node_id 视为无效，函数提前返回

## 功能列表

- ✓ 检查 NUMA 可用性
- ✓ 线程绑定到指定节点
- ✓ 查询指针所在节点
- ✓ 获取节点总数
- ✓ 内存迁移到指定节点
- ✓ 获取当前线程所在节点
