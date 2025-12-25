# c10/util/hash.h 主要功能

这个文件提供了 PyTorch c10 库的哈希工具集，包含三个核心部分：

## 1. hash_combine 函数 (46-48行)
```cpp
inline size_t hash_combine(size_t seed, size_t value) {
  return seed ^ (value + 0x9e3779b9 + (seed << 6u) + (seed >> 2u));
}
```
- 用于组合多个哈希值的实用函数
- 基于 Boost 实现，使用黄金比例常数 0x9e3779b9
- 通过位运算混合种子值和新值，避免简单的异或导致的哈希冲突

## 2. SHA1 哈希实现 (59-241行)

`sha1` 结构体实现了完整的 SHA-1 算法：

**核心方法：**
- `reset()`: 初始化 5 个 32 位哈希值为 SHA-1 标准常数
- `process_bytes()`: 处理输入字节流
- `str()`: 返回 160 位哈希的十六进制字符串表示
- `get_digest()`: 生成最终摘要，处理填充和长度追加

**算法细节：**
- 将消息分成 64 字节的块处理
- 每块扩展为 80 个 32 位字
- 使用 4 轮不同的混合函数（122-134行）
- 维护位计数器处理超长消息（166-176行）

**注意事项：**
- 代码注释明确指出 SHA-1 不再加密安全（52-53行）
- 主要用于生成唯一标识符而非安全用途

## 3. twang_mix64 函数 (243-252行)
```cpp
constexpr uint64_t twang_mix64(uint64_t key) noexcept
```
- 64 位整数的雪崩哈希函数（avalanche hash）
- 通过位移和混合操作实现良好的分布特性
- `constexpr` 允许编译期计算

## 4. c10::hash 通用哈希框架 (254-379行)

### 分发机制 (_hash_detail 命名空间)
使用 SFINAE 和模板特化实现三层分发策略：

1. **标准类型** (273-276行): 优先使用 `std::hash`
2. **枚举类型** (279-282行): 转换为底层整数类型后哈希
3. **自定义类型** (285-287行): 调用类型的静态 `hash()` 方法

### 特化实现

**std::tuple 哈希** (300-321行)
- 递归模板计算每个元素的哈希
- 使用 `hash_combine` 组合结果
- 基础情况处理单元素元组（312-316行）

**std::pair 哈希** (324-329行)
- 转换为 tuple 复用逻辑

**c10::ArrayRef 和 std::vector 哈希** (332-348行)
- 遍历元素，用 `hash_combine` 累积哈希值
- vector 直接委托给 ArrayRef 实现

**c10::complex 哈希** (372-377行)
- 组合实部和虚部的哈希值

### get_hash 便捷函数 (366-369行)
```cpp
template <typename... Types>
size_t get_hash(const Types&... args)
```
- 可变参数模板，一次哈希多个值
- 内部转换为 tuple 处理
- 简化自定义类型的哈希函数实现

## 使用示例

文件注释中的 SHA-1 用例（54-57行）：
```cpp
c10::sha1 sha1_hash{code};
const auto hash_code = sha1_hash.str();
```

推荐的自定义结构体哈希（362-365行）：
```cpp
static size_t hash(const MyStruct& s) {
  return get_hash(s.member1, s.member2, s.member3);
}
```

## 设计特点

- **类型安全**: 通过模板和 SFINAE 自动选择合适的哈希策略
- **可扩展**: 支持自定义类型通过静态 `hash()` 方法或标准接口
- **组合性**: `hash_combine` 和 `get_hash` 简化复合类型哈希
- **标准兼容**: 优先使用 `std::hash`，与 STL 容器无缝集成
