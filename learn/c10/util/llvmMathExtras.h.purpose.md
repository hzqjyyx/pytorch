这个文件提供了一系列底层位操作和数学工具函数，主要功能包括:

## 核心功能模块

### 1. 位计数操作
- **前导零计数** (`countLeadingZeros`): 从最高位开始数零的个数，利用编译器内建函数 `__builtin_clz/clzll` 或 MSVC 的 `_BitScanReverse`，fallback 到二分法实现
- **尾随零计数** (`countTrailingZeros`): 从最低位开始数零的个数，使用 `__builtin_ctz/ctzll` 或 `_BitScanForward`
- **前导/尾随1计数** (`countLeadingOnes`/`countTrailingOnes`): 通过取反后调用零计数实现
- **总1计数** (`countPopulation`): 统计所有设置为1的位数，使用 `__builtin_popcount/popcountll` 或位操作算法

### 2. 位查找操作
- `findFirstSet`: 找到第一个被设置的位的索引
- `findLastSet`: 找到最后一个被设置的位的索引

### 3. 位掩码生成
- `maskTrailingOnes/maskLeadingOnes`: 生成右/左侧N位为1的掩码
- `maskTrailingZeros/maskLeadingZeros`: 生成右/左侧N位为0的掩码

### 4. 对数运算
- `Log2`: 浮点数的log2运算（Android API < 18有特殊处理）
- `Log2_32/Log2_64`: 整数的向下取整log2（通过前导零计数实现）
- `Log2_32_Ceil/Log2_64_Ceil`: 整数的向上取整log2

### 5. 2的幂相关
- `isPowerOf2_32/isPowerOf2_64`: 判断是否为2的幂
- `NextPowerOf2`: 返回下一个严格大于给定值的2的幂
- `PowerOf2Floor/PowerOf2Ceil`: 向下/向上取整到2的幂

### 6. 整数范围检查
- `isInt<N>`: 检查有符号整数是否能用N位表示（包含特化版本：8/16/32位）
- `isUInt<N>`: 检查无符号整数是否能用N位表示
- `isShiftedInt/isShiftedUInt`: 检查是否为N位数左移S位
- `isIntN/isUIntN`: 动态位宽版本
- `minIntN/maxIntN/maxUIntN`: 获取N位整数的最值

### 7. 掩码模式识别
- `isMask_32/isMask_64`: 判断是否为从最低位开始的连续1
- `isShiftedMask_32/isShiftedMask_64`: 判断是否包含连续的1序列

### 8. 位反转
- `reverseBits`: 使用查找表 `BitReverseTable256` 实现字节级别的位反转

### 9. 64位操作
- `Hi_32/Lo_32`: 提取64位整数的高/低32位
- `Make_64`: 从两个32位整数组合成64位整数

### 10. 对齐操作
- `alignTo`: 将值向上对齐到指定倍数（支持偏移量Skew）
- `alignDown`: 将值向下对齐
- `alignAddr`: 将地址对齐到指定字节边界
- `alignmentAdjustment`: 计算对齐所需的调整量
- `OffsetToAlignment`: 计算到下一个对齐点的偏移
- `MinAlign`: 计算两个对齐值/偏移量的最小公共对齐
- `divideCeil`: 整数除法向上取整

### 11. 符号扩展
- `SignExtend32/SignExtend64`: 将低B位符号扩展到32/64位（模板版本和运行时版本）

### 12. 饱和运算
- `SaturatingAdd`: 带溢出检测的饱和加法
- `SaturatingMultiply`: 带溢出检测的饱和乘法
- `SaturatingMultiplyAdd`: 带溢出检测的饱和乘加运算

### 13. 类型转换工具
- `BitsToDouble/BitsToFloat`: 将整数按位模式转换为浮点数
- `DoubleToBits/FloatToBits`: 将浮点数转换为整数位模式（使用 `memcpy` 避免UB）

### 14. 其他工具
- `AbsoluteDifference`: 无符号整数的绝对差值
- `GreatestCommonDivisor64`: 最大公约数（欧几里得算法）

## 实现特点
- 使用模板特化针对不同大小类型优化（4字节/8字节）
- 优先使用编译器内建函数（GCC/Clang的 `__builtin_*`，MSVC的 `_BitScan*`）
- 提供 fallback 的纯C++实现（二分法、位操作算法）
- 大量使用 `constexpr` 支持编译期计算
- 包含特殊平台处理（Android NDK、MSVC警告抑制）

---

**ROCm/Backward相关**:
- 无ROCm特定内容
- 无向后兼容性问题（来自LLVM项目，保持稳定接口）
