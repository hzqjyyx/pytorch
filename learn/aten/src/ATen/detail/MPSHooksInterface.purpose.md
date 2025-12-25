# MPSHooksInterface 文件分析

## 核心功能

**MPSHooksInterface** 是 PyTorch 中用于 Apple Metal Performance Shaders (MPS) 后端的接口定义和工厂实现。它采用**注册表模式（Registry Pattern）**，允许在编译时动态加载 MPS 实现。

## 文件职责

### MPSHooksInterface.h
- 定义 `MPSHooksInterface` 结构体，继承自 `AcceleratorHooksInterface`
- 提供一套虚函数接口，覆盖 MPS 后端的所有功能
- 使用 `FAIL_MPSHOOKS_FUNC` 宏为所有函数提供默认实现（抛出异常），用于当 MPS 后端未可用时的错误提示
- 声明注册表类型 `MPSHooksRegistry`

### MPSHooksInterface.cpp
- 实现 `getMPSHooks()` 函数，用于获取 MPS hooks 的单例
- 在非移动平台上，尝试从注册表创建真实的 `MPSHooks` 实现
- 若注册表中无可用实现，回退到默认的 `MPSHooksInterface`（会在调用任何方法时抛出异常）
- 定义全局注册表 `MPSHooksRegistry`

## 接口涵盖的功能

- **初始化**: `init()`、`hasMPS()`、macOS 版本检查
- **内存管理**: 内存分配器、缓存管理、内存使用统计
- **生成器**: 默认和新建随机数生成器
- **同步**: 设备同步、流提交
- **事件管理**: 获取/释放/记录/等待事件、事件时间差计算
- **性能分析**: Profiler 追踪启动/停止
- **其他**: 命令缓冲区、分发队列、内存比例设置、固定内存检查

## 设计模式

- **工厂模式**: `getMPSHooks()` 作为工厂函数
- **注册表模式**: 允许插件式注册 MPS 实现
- **单例模式**: 静态变量确保全局唯一的 hooks 实例
- **防御性编程**: 默认实现返回错误，而非无声失败

## 关键特点

- **条件编译**: `C10_MOBILE` 宏控制移动平台行为
- **延迟初始化**: hooks 在首次调用时创建
- **优雅降级**: 无可用 MPS 后端时提供清晰错误信息

## 快速总结

- **MPSHooksInterface.h** → MPS 后端功能接口定义 + 注册表声明
- **MPSHooksInterface.cpp** → MPS hooks 单例工厂实现 + 注册表定义
- **作用**: 为 PyTorch 提供可插拔的 Apple Metal GPU 后端支持
- **关键函数**: `getMPSHooks()` 获取全局 MPS 接口实例
- **核心机制**: 运行时注册表查询，动态加载真实实现或返回错误桩
