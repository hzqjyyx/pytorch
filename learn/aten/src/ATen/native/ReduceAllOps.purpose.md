## ReduceAllOps.h

声明了两个函数指针类型和两个调度存根：
- `reduce_all_fn`: 接收结果张量和输入张量，执行规约操作
- `reduce_min_max_fn`: 同时处理最大值和最小值的规约
- `min_all_stub` 和 `max_all_stub`: 调度器，根据设备类型调用相应实现

## ReduceAllOps.cpp

实现了四个核心函数：

### min(const Tensor &self)
- 检查张量元素数量 > 0
- 创建标量张量作为结果
- 调用 `min_all_stub` 执行最小值规约

### min_unary_out(const Tensor &self, Tensor& out)
- 验证输入输出设备相同
- 验证数据类型可转换
- 重新调整输出张量尺寸为标量
- 执行规约并存储到输出张量

### max(const Tensor &self)
- 与 min 逻辑相同，执行最大值规约

### max_unary_out(const Tensor &self, Tensor& out)
- 与 min_unary_out 逻辑相同，执行最大值规约

### _aminmax_all(const Tensor &self)
- 已弃用的函数
- 发出一次警告后委托给 `at::aminmax`

---

- **核心功能**: 实现张量的全局规约操作（最小值、最大值）
- **设计模式**: 使用调度器（stub）支持多设备（CPU/GPU）
- **API 分类**: 创建新结果张量 vs 写入已有张量两种形式
- **验证机制**: 设备检查、dtype 兼容性验证、张量元素数量检查
- **连续性**: 规约前强制张量连续化（`.contiguous()`）
