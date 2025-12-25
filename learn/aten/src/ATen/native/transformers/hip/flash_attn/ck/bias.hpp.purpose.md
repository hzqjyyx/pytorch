这个文件定义了用于Flash Attention在ROCm/HIP上的偏置(Bias)处理机制。

**主要功能：**

- **bias_enum枚举**：定义三种偏置类型
  - `no_bias`：无偏置
  - `elementwise_bias`：逐元素偏置
  - `alibi`：ALiBi (Attention with Linear Biases)偏置

- **bias_info结构体**：包含偏置类型和秩信息(rank_info)
  - 用于区分偏置的维度布局(1×1×s×s / 1×h×s×s / b×h×s×s 等)

- **serialize()方法**：将偏置信息序列化为字符串
  - "n"表示no_bias
  - "e"表示elementwise_bias
  - "alibi"表示alibi类型
  - 可选的[rank]后缀表示秩信息

- **decode()静态方法**：从字符串反序列化偏置信息
  - 支持多种输入格式("0"/"n", "1"/"e"/"elementwise", "2"/"a"/"alibi")
  - 解析":"分隔符后的秩信息

- **流操作符**：支持通过`<<`将bias_info输出到ostream

**核心作用**：
- 提供统一的偏置配置表示方法
- 便于Attention计算中的偏置应用和参数化
- 支持序列化/反序列化用于内核选择或日志记录
