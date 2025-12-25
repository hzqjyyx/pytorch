**主要功能：CUDA 扫描（前缀和）工具库**

- **inclusiveBinaryPrefixScan** (15-59行)：实现包含性二进制前缀扫描
  - 使用 warp voting 进行线程内扫描
  - 通过共享内存完成跨 warp 的扫描
  - 支持自定义二元操作符

- **exclusiveBinaryPrefixScan** (64-76行)：实现排除性二进制前缀扫描
  - 基于 inclusiveBinaryPrefixScan 实现
  - 通过减去当前元素值转换为排除性扫描
  - 返回传出的进位值供后续计算使用

- **核心机制**
  - 第一阶段：利用 `WARP_BALLOT` 和 `__popc` 进行 warp 内投票计数
  - 第二阶段：单线程在共享内存中扫描各 warp 的结果
  - 第三阶段：各线程加载前一个 warp 的累积值

- **优化特性**
  - KillWARDependency 模板参数：控制是否添加额外同步点解决 WAR 依赖
  - 针对 CC 3.0+ 架构设计，单线程扫描比 warp shuffle 扫描更高效
