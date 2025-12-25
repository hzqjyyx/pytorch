# LeftRight 同步原语

这是一个实现了 wait-free readers 的并发数据结构同步机制。基于论文 https://hal.archives-ouvertes.fr/hal-01207881/document

## 核心机制

**双副本设计**：维护两份相同的数据副本（Left/Right 或 A/B）
- 一份为前台（foreground），供读者访问
- 一份为后台（background），供写者修改

**读操作**（c10/util/LeftRight.h:80-85）：
- 原子递增对应的读计数器（通过 IncrementRAII RAII 包装）
- 读取当前前台数据
- 原子递减计数器
- **不需要锁**，实现 wait-free 读取

**写操作**（c10/util/LeftRight.h:91-95, 99-162）：
1. 加互斥锁保证写的互斥
2. 写入后台副本（A）
3. 切换前台数据指针（A↔B）
4. 等待旧计数器归零（确保旧读者全部完成）
5. 切换前台计数器指针
6. 等待新计数器归零（确保间隙期间的读者完成）
7. 写入新后台副本（B）

**容错机制**（c10/util/LeftRight.h:164-176）：
- 如果写函数抛异常，自动从前台副本复制恢复后台状态

## 辅助类

**IncrementRAII**（c10/util/LeftRight.h:12-28）：
- 原子计数器的 RAII 封装
- 构造函数递增，析构函数递减
- 防止手动计数错误

**RWSafeLeftRightWrapper**（c10/util/LeftRight.h:193-223）：
- API 兼容的简化版本
- 用读写锁替代 LeftRight
- 性能较低但实现简单

## 性能特征

- **读优化**：wait-free，但需要一次原子写操作
- **写成本高**：每次写执行两遍（维护两个副本）
- **适用场景**：读远多于写的场景

## 要点

- 不可复制/移动（禁用拷贝和移动构造）
- 析构函数等待所有读写操作完成
- 解耦读写的阻塞关系，实现读的无锁化
- PyTorch 代码中使用较少
