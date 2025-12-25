## SobolEngineOpsUtils 文件功能分析

这两个文件为 PyTorch 的 **Sobol 准随机序列生成器** 提供核心工具函数和预计算常量数据。

### 头文件 (SobolEngineOpsUtils.h)

**位运算辅助函数：**
- `bit_length(n)` - 计算表示整数 `n` 所需的最小位数
- `rightmost_zero(n)` - 找到整数二进制表示中最右边的 0 的位置（零索引）
- `bitsubseq(n, pos, length)` - 从整数的二进制表示中提取从 `pos` 开始、长度为 `length` 的子序列

**张量运算函数：**
- `cdot_pow2(bmat)` - 对批量方阵与 2 的幂向量进行内积运算

**预定义常量：**
- `MAXDIM = 21201` - 支持的最大维度数
- `MAXDEG = 18` - 最大多项式度数
- `MAXBIT = 30` - 最大位数
- `LARGEST_NUMBER = 2^30` - 最大数值
- `RECIPD = 1.0 / 2^30` - 用于归一化到 [0,1) 区间

### 源文件 (SobolEngineOpsUtils.cpp)

该文件包含两个巨大的预计算数据数组（约 2.1MB）：
- `poly[21201]` - 本原多项式系数数组
- `initsobolstate[21201][18]` - Sobol 序列初始化状态矩阵

数据来源于 S. Joe 和 F. Y. Kuo 的论文 *"Remark on algorithm 659: Implementing Sobol's quasirandom sequence generator"* (ACM TOMS, 2003)，从 UNSW 大学网站的方向数文件生成。

---

**要点总结：**

- 提供 Sobol 准随机数生成器所需的位操作工具函数
- 支持最高 21201 维的 Sobol 序列生成
- 预存储本原多项式 (`poly`) 和初始方向数 (`initsobolstate`)
- `cdot_pow2` 用于将二进制方向向量转换为十进制数值
- 常量 `RECIPD` 将整数结果归一化到 [0, 1) 浮点区间
- 数据基于 Joe-Kuo 方向数，这是目前最广泛使用的高质量 Sobol 方向数
