## CPUGeneratorImpl 核心功能

这是 PyTorch 中 CPU 平台随机数生成器的实现，负责管理和维护伪随机数生成状态。

### 主要组件

**1. 随机数生成引擎**
- 使用 MT19937（Mersenne Twister）算法作为底层引擎
- 引擎状态包含：种子值、状态数组（624个32位无符号整数）、位置指针等
- 提供 `random()` 生成32位随机数，`random64()` 通过两次调用合成64位随机数

**2. 种子管理**
- `set_current_seed()`: 手动设置种子，会重置引擎和缓存的正态分布样本
- `current_seed()`: 获取当前种子
- `seed()`: 从 `/dev/urandom` 或时间获取非确定性随机数作为种子

**3. 正态分布采样缓存**
- 缓存两种精度的正态分布样本：`next_float_normal_sample_` 和 `next_double_normal_sample_`
- Box-Muller 变换一次生成两个正态分布样本，缓存机制避免浪费第二个样本
- 提供 getter/setter 方法供分布采样器使用

**4. 状态序列化/反序列化**

支持两种状态格式以保持向后兼容：

- **Legacy 格式** (`CPUGeneratorImplStateLegacy`)：
  - 存储64位状态数组
  - 保存正态分布的中间值（normal_x, normal_y, normal_rho）
  
- **Current 格式** (`CPUGeneratorImplState`)：
  - 包含 legacy 部分加上 float 精度的正态分布样本
  - 直接存储计算好的正态分布样本，而非中间值

`set_state()` 根据输入张量大小自动识别格式，`get_state()` 始终返回新格式。

### 设计要点

**全局默认生成器**
```cpp
const Generator& getDefaultCPUGenerator() {
  static auto default_gen_cpu = createCPUGenerator(c10::detail::getNonDeterministicRandom());
  return default_gen_cpu;
}
```
维护全局单例，用于未显式指定生成器的操作。

**不支持 offset**
- `set_offset()` 和 `get_offset()` 直接抛出错误
- offset 机制主要用于 CUDA 并行生成，CPU 不需要此功能

**克隆机制**
- `clone()` 创建深拷贝，复制引擎状态和缓存样本
- 用于需要独立随机数流的场景

---

**ROCm/Backward 相关**（简要）：
- 无 ROCm 特定代码
- 状态反序列化支持读取旧版检查点（backward compatibility）
- Legacy 状态格式的支持未来可能移除（注释标注 FIXME）
