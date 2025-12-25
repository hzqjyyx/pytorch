# LegacyVmapTransforms 核心功能解析

## 主要用途

这些文件负责 PyTorch 的 `vmap` 功能中的张量视图转换，将用户视角的**逻辑张量**转换为底层实际的**物理张量**，以及反向转换。

## 关键概念

**逻辑视图 vs 物理视图**：
- 逻辑视图：用户在 `vmap` 内部看到的张量形状（如 `[3]`）
- 物理视图：底层实际存储的张量形状（如 `[2, 3, 4]`，其中 2 和 4 是批次维度）

示例：
```python
vmap(vmap(func, in_dims=(2,)), in_dims=(0,))(torch.ones(2, 3, 4))
# 产生 BatchedTensor(ones(2, 3, 4), bdims=[(lvl=1,dim=0),(lvl=2,dim=2)])
# 逻辑形状: [3]
# 物理形状: [2, 3, 4]
```

## 核心组件

### 1. VmapPhysicalView (LegacyVmapTransforms.h:114-154)

表示物理张量视图的数据结构：
- 存储物理张量（所有批次维度已移到前面）
- 通过 `levels_` 位集记录哪些 vmap 层级对应批次维度
- 提供逻辑到物理的映射方法

**关键方法**：
- `getPhysicalDims()`: 将逻辑维度索引转为物理维度索引（加上批次维度偏移）
- `getPhysicalShape()`: 将逻辑形状转为物理形状（前置批次大小）
- `numBatchDims()`: 返回批次维度数量

### 2. MultiBatchVmapTransform (LegacyVmapTransforms.h:58-61)

处理多批次维度的转换器。

**单张量转换**（LegacyVmapTransforms.cpp:43-49）：
1. 提取 BatchedTensor 的批次维度信息
2. 调用 `permuteBatchDimsToFront()` 将批次维度移到前面
3. 返回 VmapPhysicalView

**多张量转换**（LegacyVmapTransforms.cpp:189-236）：
1. 收集所有张量的集体 vmap 层级
2. 对每个张量调用 `alignBatchDimsAtFront()` 对齐批次维度（缺失的层级补 size=1 的维度）
3. 计算统一的批次大小（取各张量对应维度的最大值）
4. 扩展所有张量到统一的批次大小

### 3. BroadcastingVmapTransform (LegacyVmapTransforms.h:82-84)

专门用于广播操作的转换器（LegacyVmapTransforms.cpp:256-280）：
1. 找到所有张量的集体层级和最大逻辑维度数
2. 对齐批次维度到前面
3. 对齐非批次维度（在批次维度和逻辑维度之间插入 size=1 的维度）

示例：
- 输入：`(B, 2)` 和 `(B, 3, 2)` → 输出：`(B, 1, 2)` 和 `(B, 3, 2)`
- 输入：`(B, 2)` 和 `(2,)` → 输出：`(B, 2)` 和 `(1, 2)`

### 4. VmapPhysicalToLogicalMap (LegacyVmapTransforms.h:160-181)

反向映射：将物理张量转回逻辑 BatchedTensor。

**apply() 方法**（LegacyVmapTransforms.cpp:286-288）：
- 假设物理张量的前 N 个维度是批次维度
- 根据 `levels_` 位集创建 BatchDims
- 调用 `makeBatched()` 构造 BatchedTensor

## 核心算法

### permuteBatchDimsToFront (LegacyVmapTransforms.cpp:20-41)

将批次维度重排到张量前面：
1. 检查批次维度是否已按顺序在前面
2. 构造排列向量：先放批次维度位置，再放非批次维度位置
3. 调用 `permute()` 重排

### alignBatchDimsAtFront (LegacyVmapTransforms.cpp:134-179)

对齐批次维度到指定层级集合：
1. 获取张量的物理形式和现有层级
2. 计算当前的逻辑维度数
3. 构造对齐后的形状向量（初始化为 1）
4. 从右向左复制逻辑维度的大小
5. 从左向右填充批次维度的大小（缺失的层级保持 size=1）
6. 调用 `view()` 重塑

## 数据类型定义

- `VmapDimVector`: 小向量优化的维度数组（静态大小 8）
- `VmapPhysicalViewVec`: 小向量优化的 VmapPhysicalView 数组（静态大小 4）
- `kVmapNumLevels`: vmap 嵌套层级的最大数量（通过位集管理）

## 辅助函数

- `areBdimsAtFrontInOrder()`: 检查批次维度是否已按顺序排列（LegacyVmapTransforms.cpp:9）
- `createBatchDimBitset()`: 创建批次维度位置的位集（未在文件中定义，外部引用）
- `createVmapLevelsBitset()`: 从 BatchDims 创建层级位集（未在文件中定义，外部引用）
- `computeFrontBatchDimsFromLevels()`: 从层级位集计算前置的 BatchDims（LegacyVmapTransforms.cpp:91）
- `getPhysicalTensorAndLevels()`: 获取物理张量和层级信息（LegacyVmapTransforms.cpp:106）

---

**不相关内容**：无 ROCm 或 Backward 相关代码
