MT19937RNGEngine.h 实现了一个 Mersenne Twister (MT19937) 伪随机数生成器。这是 PyTorch 的核心随机数引擎。

**主要组件：**

- **常量定义**：状态数组大小（N=624）、中间索引（M=397）、变换矩阵常数等
- **mt19937_data_pod 结构体**：存储 RNG 的完整状态，包括种子、左侧计数、状态数组等，用于状态的序列化/反序列化
- **mt19937_engine 类**：核心随机数生成引擎
  - 构造函数通过 64 位种子初始化
  - `operator()()`：生成下一个 32 位随机数，包含初始化检查和 tempering 变换
  - `init_with_uint32()`：初始化状态数组
  - `next_state()`：生成新的 624 个状态值（Mersenne Twister 的核心算法）
  - `mix_bits()` 和 `twist()`：辅助函数处理位操作

**设计决策：**

- 自定义实现而非 `std::mt19937`，原因是在 PyTorch 的 -O2 优化下，配合 `at::uniform_real_distribution` 时性能更好
- 使用 32 位而非 64 位状态数组来优化性能
- 避免 `std::uniform_real_distribution` 的已知 bug（LWG #2524）

**关键特性：**

- 支持状态保存/恢复（`data()`、`set_data()`）
- 状态有效性检验（`is_valid()`）
- 完全兼容标准 Mersenne Twister 算法
