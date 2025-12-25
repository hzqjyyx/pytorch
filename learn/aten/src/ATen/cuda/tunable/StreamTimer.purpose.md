## StreamTimer 功能分析

`StreamTimer` 和 `StreamTimerNoSync` 是 CUDA 流事件计时器的实现，用于测量 CUDA 操作的执行时间。

### StreamTimer（同步版本）

在 `StreamTimer.h:18-32` 中定义，`StreamTimer.cpp:19-41` 中实现：

- **构造/析构**：创建和销毁 CUDA 事件对象（`cudaEvent_t`）
- **Start()**：同步 start 事件，然后在当前 CUDA 流上记录时间戳
- **End()**：在当前 CUDA 流上记录结束时间戳，然后同步 end 事件
- **Duration()**：计算两个事件之间的经过时间（毫秒级，微秒精度）

### StreamTimerNoSync（异步版本）

在 `StreamTimer.h:34-48` 中定义，`StreamTimer.cpp:43-64` 中实现：

- **构造/析构**：同样创建和销毁 CUDA 事件对象
- **Start()**：直接记录 start 事件，不进行同步
- **End()**：直接记录 end 事件，不进行同步
- **Duration()**：仅在计算前同步 end 事件，然后计算经过时间

### 关键区别

- `StreamTimer` 在 Start 和 End 都进行同步，确保精确计时但开销更大
- `StreamTimerNoSync` 减少同步调用，仅在获取结果时同步一次，性能开销更小

### 主要功能点

- 基于 CUDA 事件的高精度计时器
- 支持两种计时策略（完全同步 vs 最小同步）
- 返回经过时间以毫秒为单位
- 用于 PyTorch 的可调优操作（TunableOp）性能测量
