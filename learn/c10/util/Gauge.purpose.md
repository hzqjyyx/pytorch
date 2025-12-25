## Gauge 系统概述

这是 PyTorch 中的**监控指标收集系统**，用于记录运行时的数值指标（如内存使用、操作计数等）。

## 核心设计

**分层架构**：
- `GaugeBackendIf`：后端接口，定义具体指标记录的行为
- `GaugeBackendFactoryIf`：后端工厂接口，负责创建后端实例
- `GaugeImpl`：内部实现类，管理所有已注册的后端
- `GaugeHandle`：公开API，用户通过此类记录指标

**关键特性**：

1. **单例模式**（getInstance）
   - 使用静态 map 存储不同 key 的 GaugeImpl 实例
   - 线程安全：通过 `Synchronized` 包装实现互斥访问

2. **后端注册机制**（registerGaugeBackend）
   - 允许动态注册多个后端工厂
   - 创建 GaugeImpl 时，遍历所有工厂创建对应后端
   - 后端可选择性忽略某些指标（factory 返回 nullptr）

3. **多后端支持**（SmallVector<backends_>）
   - 记录时调用所有已注册后端
   - 支持同时向多个目标写入指标（如本地存储、远程服务等）

4. **宏便利接口**（STATIC_GAUGE）
   - 便捷创建静态 Gauge 句柄
   - 示例：`STATIC_GAUGE(memory_allocated).record(value)`

## 主要功能点

- **指标记录**：GaugeHandle::record() 分发记录请求到所有后端
- **动态后端注册**：支持运行时添加新的监控后端而无需修改核心代码
- **线程安全**：使用 Synchronized 保护共享数据结构
- **懒加载**：首次使用某个 key 时才创建对应的 GaugeImpl 实例
- **工厂模式**：解耦指标收集点与具体后端实现
