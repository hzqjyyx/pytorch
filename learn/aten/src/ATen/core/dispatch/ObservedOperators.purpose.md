# ObservedOperators 文件分析

## 功能概述

`ObservedOperators` 是一个用于管理操作符观察/追踪的静态工具类，用于区分哪些操作符应该被观察（profiled/traced），哪些不应该被观察。

## 详细说明

**header 文件** (`ObservedOperators.h`)：
- 定义了一个不可实例化的结构体 `ObservedOperators`（删除了默认构造函数）
- 提供两个静态方法的声明：
  - `isObserved()`：检查某个操作符是否应该被观察
  - `getUnobservedOperatorList()`：获取不应该被观察的操作符列表

**implementation 文件** (`ObservedOperators.cpp`)：
- `getUnobservedOperatorList()` 返回一个静态的不可观察操作符集合，包含：
  - 张量元信息查询操作：`aten::size`、`aten::is_leaf`、`aten::output_nr`、`aten::_version`、`aten::is_complex`
  - profiler 内部操作：`profiler::_record_function_enter`、`profiler::_record_function_enter_new`、`profiler::_record_function_exit`

- `isObserved()` 检查操作符名称是否在不可观察列表中，如果不在列表中则认为该操作符应该被观察

## 核心要点

- **目的**：过滤掉不需要性能分析/追踪的低级操作符
- **设计模式**：使用黑名单机制（unobserved list）
- **应用场景**：profiler 和动态追踪系统中，避免对元信息查询和 profiler 自身操作的重复追踪

## Bullet Points

- 提供操作符可观察性检查的静态接口
- 维护黑名单列表，包含不需要被观察的操作符
- 包含张量元信息查询操作（size、is_leaf 等）和 profiler 内部操作
- 使用 unordered_set 进行高效的成员查询
- 设计为纯工具类，不允许实例化
