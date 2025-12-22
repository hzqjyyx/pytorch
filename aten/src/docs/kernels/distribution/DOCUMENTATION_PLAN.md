# PyTorch 随机数系统文档规划

> **面向**: 文档写作者和维护者
> **目的**: 记录整个文档体系的规划、设计原则和写作指南

---

## 文档体系设计：方案一 + 总览文档

### 核心设计理念

采用 **"按分布类型分文档 + 总览文档"** 的组织方式，具体原因如下：

1. **模块化**: 每个文档聚焦一个主题，长度适中（500-1000 行），便于阅读和维护
2. **按需查阅**: 读者可以根据需要直接跳转到特定分布的文档
3. **避免冗长**: 单个文档不会超过 1500 行，避免阅读疲劳
4. **便于扩展**: 新增分布时只需添加新文档，不影响现有结构
5. **复用结构**: 所有详细文档可以复用 `uniform_call_flow.md` 的结构模板

---

## 文档结构清单

### 已完成文档 ✅

| 文件名 | 状态 | 内容概要 | 行数 | 完成度 |
|--------|------|----------|------|--------|
| `DOCUMENTATION_PLAN.md` | ✅ 已完成 | 文档规划（面向写作者） | 本文档 | 100% |
| `architecture.md` | ✅ 已完成 | 总览文档（面向读者） | ~500 行 | 100% |
| `uniform_call_flow.md` | ✅ 已完成 | torch.rand/uniform 详解（CUDA 重点） | ~800 行 | 100% |
| `normal_call_flow.md` | ✅ 已完成 | torch.randn/normal 详解（CUDA 重点） | ~1050 行 | 100% |
| `discrete_call_flow.md` | ✅ 已完成 | torch.randint/random 详解 | ~1600 行 | 100% |

### 待编写文档 📝

| 文件名 | 优先级 | 预估行数 | 目标读者 | 主要内容 |
|--------|--------|----------|----------|----------|
| `special_distributions.md` | ⭐ 低 | ~800 行 | 高级用户 | exponential, cauchy 等特殊分布 |
| `sampling.md` | ⭐ 低 | ~900 行 | 高级用户 | multinomial, poisson 等采样函数 |

---

## 详细文档规划

### 1. normal_call_flow.md（高优先级）

**为什么优先**：
- `torch.randn` 是第二常用的随机数 API（仅次于 `torch.rand`）
- 正态分布在深度学习中应用最广（权重初始化、噪声注入等）
- 可以直接复用 `uniform_call_flow.md` 的结构

**文档大纲**：

```markdown
# torch.randn/normal 正态分布随机数生成详解

## 1. Python 层入口
- torch.randn vs torch.normal vs tensor.normal_()
- API 签名对比

## 2. C++ 分派层
- torch.randn 入口函数 (TensorFactories.cpp)
- 带生成器的版本
- 输出参数版本

## 3. 核心分布层
- normal_ 实现 (Distributions.cpp:275)
- normal_impl_ 模板函数
- TensorIterator 创建

## 4. CPU 实现（简介）
- Box-Muller 变换原理
- 为什么缓存一个样本（Box-Muller 一次生成两个）

## 5. CUDA 实现（重点）
### 5.1 CUDA normal_kernel 入口
### 5.2 CUDA 模板实现
### 5.3 normal_and_transform 实现
### 5.4 distribution_nullary_kernel（复用）
### 5.5 CUDA Kernel 执行

## 6. Box-Muller 算法原理
- 基本 Box-Muller 变换
- Marsaglia 极坐标形式（可能的优化）
- cuRAND 的 curand_normal4 实现

## 7. 完整调用流程图（CUDA）
- 类似 uniform_call_flow.md 的详细流程

## 8. 与 uniform 的对比
- 实现差异
- 性能差异
- 使用场景差异

## Appendix A: Box-Muller 变换详解
## Appendix B: 正态分布的统计特性
```

**关键实现细节**：

1. **Box-Muller 变换**：
   ```cpp
   // 基本形式
   U1, U2 ~ Uniform(0, 1)
   Z1 = sqrt(-2 * log(U1)) * cos(2π * U2)
   Z2 = sqrt(-2 * log(U1)) * sin(2π * U2)
   Z1, Z2 ~ Normal(0, 1)
   ```

2. **cuRAND 实现**：
   - `curand_normal4()` 一次生成 4 个正态随机数
   - 内部使用 Box-Muller + 优化

3. **与 uniform 的对比**：
   | 特性 | uniform | normal |
   |------|---------|--------|
   | 变换 | 线性映射 | Box-Muller |
   | cuRAND 函数 | curand_uniform4 | curand_normal4 |
   | 计算复杂度 | O(1) | O(log + trig) |

---

### 2. discrete_call_flow.md（中优先级）

**文档大纲**：

```markdown
# torch.randint/random 离散整数随机数生成详解

## 1. Python 层入口
- torch.randint API
- torch.randperm API
- tensor.random_() API

## 2. C++ 分派层
- randint 入口函数
- randperm 入口函数
- random_ 的三种重载

## 3. 核心实现层
### 3.1 random_ 实现
- random_impl_ 模板函数
- 整数范围处理
- 数据类型检查

### 3.2 randperm 实现
- 算法选择（小 n vs 大 n）
- CPU: std::shuffle
- CUDA: thrust::shuffle

## 4. CUDA 实现
### 4.1 uniform 到整数的转换
### 4.2 边界处理
- 避免 modulo bias
- 使用拒绝采样

## 5. 完整调用流程图

## Appendix A: Modulo Bias 问题
## Appendix B: Fisher-Yates Shuffle 算法
```

**关键实现细节**：

1. **uniform 转整数**：
   ```cpp
   // 简单但有偏差的方法（不要用）
   int x = (int)(uniform() * range);  // ❌ modulo bias

   // 正确方法：拒绝采样
   uint64_t range = high - low;
   uint64_t limit = UINT64_MAX - (UINT64_MAX % range);
   uint64_t r;
   do {
       r = uniform_uint64();
   } while (r >= limit);
   return low + (r % range);
   ```

2. **randperm 算法**：
   - 小 n (< 10000): 生成数组再 shuffle
   - 大 n: 使用 reservoir sampling 或 Floyd 算法

---

### 3. special_distributions.md（低优先级）

**文档大纲**：

```markdown
# PyTorch 特殊分布实现详解

## 1. 逆变换采样（Inverse Transform Sampling）
### 1.1 理论基础
- CDF 逆变换方法
- 为什么有效

### 1.2 Exponential 分布
- tensor.exponential_(lambda)
- 实现: -log(1 - U) / lambda
- CUDA 实现

### 1.3 Cauchy 分布
- tensor.cauchy_(median, sigma)
- 实现: median + sigma * tan(π(U - 0.5))
- 重尾分布特性

### 1.4 Geometric 分布
- tensor.geometric_(p)
- 实现: ⌈log(U) / log(1-p)⌉
- 离散分布的逆变换

## 2. 复合分布
### 2.1 Log-Normal 分布
- tensor.log_normal_(mean, std)
- 实现: exp(normal(mean, std))
- 两步变换

## 3. Bernoulli 分布
### 3.1 tensor.bernoulli_(p)
- 实现: uniform() < p
- 向量化实现

## 4. 实现对比表
- 各分布的算法、复杂度、应用场景

## Appendix: 逆变换采样的数学证明
```

**关键内容**：

1. **逆变换采样原理**：
   ```
   若 U ~ Uniform(0,1)，F 是累积分布函数
   则 X = F^(-1)(U) 服从分布 F
   ```

2. **各分布的变换公式**：
   - Exponential: `X = -ln(1-U) / λ`
   - Cauchy: `X = μ + σ * tan(π(U-0.5))`
   - Geometric: `X = ⌈ln(U) / ln(1-p)⌉`

---

### 4. sampling.md（低优先级）

**文档大纲**：

```markdown
# PyTorch 复杂采样函数详解

## 1. Multinomial 采样
### 1.1 torch.multinomial API
- 用途：分类采样、语言模型
- 参数：weights, num_samples, replacement

### 1.2 算法实现
- 累积概率法（小规模）
- Alias Method（大规模）
- CUDA 并行实现

## 2. Poisson 分布
### 2.1 torch.poisson API
- 参数：lambda 张量

### 2.2 算法选择
- 小 λ: Knuth 算法
- 大 λ: 变换拒绝采样（PTRS）

## 3. Binomial 分布
### 3.1 算法
- 小 n*p: 二项式反演
- 大 n*p: BTRS 算法

## 4. Gamma 分布
### 4.1 torch._standard_gamma
### 4.2 Marsaglia-Tsang 算法
- 拒绝采样
- 接受率优化

## 5. Dirichlet 分布
### 5.1 实现：多个 Gamma 归一化

## Appendix A: Alias Method 详解
## Appendix B: BTRS 算法推导
```

**关键算法**：

1. **Multinomial - Alias Method**：
   - O(n) 预处理，O(1) 采样
   - 适合大规模重复采样

2. **Poisson - Knuth 算法**（小 λ）：
   ```python
   L = exp(-λ)
   k = 0
   p = 1
   while p > L:
       p *= uniform()
       k += 1
   return k - 1
   ```

3. **Gamma - Marsaglia-Tsang**：
   - 基于拒绝采样
   - 接受率 > 95%

---

## 文档写作指南

### 1. 文档结构模板

每个详细调用流程文档应遵循以下结构：

```markdown
# [API名称] [分布名称]随机数生成详解

## 概述
- API 简介
- 使用场景
- 与其他 API 的关系

## 1. Python 层入口
### 1.1 用户 API
### 1.2 API 绑定

## 2. C++ 分派层
### 2.1 入口函数
### 2.2 带生成器的版本
### 2.3 输出参数版本

## 3. 核心分布层
### 3.1 [distribution]_ 实现
### 3.2 [distribution]_impl_ 模板函数
### 3.3 关键功能点

## 4. CPU 实现（简介）
- 算法描述
- 关键代码片段

## 5. CUDA 实现（重点）
### 5.1 CUDA kernel 入口
### 5.2 CUDA 模板实现
### 5.3 核心算法
### 5.4 Grid-Stride Loop（可复用）
### 5.5 执行策略计算（可复用）

## 6. [特定算法] 原理
- 算法的数学原理
- 为什么这样实现

## 7. 完整调用流程图（CUDA）
- ASCII 流程图
- 标注 Generator 使用位置

## 8. [可选] 与其他分布的对比
- 实现差异
- 性能对比
- 使用建议

## Appendix A: [算法名称] 详解
## Appendix B: [可选的扩展内容]
```

### 2. 代码片段格式

**C++ 代码**：
```cpp
// 文件位置: aten/src/ATen/native/XXX.cpp:行号

函数签名 {
  // 注释说明关键步骤
  代码实现
}
```

**CUDA Kernel**：
```cpp
// 文件位置: aten/src/ATen/native/cuda/XXX.cu:行号

__global__ void kernel_name(...) {
  // 注释说明
  代码实现
}
```

### 3. 流程图格式

使用 ASCII 流程图，参考 `uniform_call_flow.md` 的风格：

```
torch.randn(3, 4, device='cuda')
    ↓
torch._C.randn
    ↓
at::randn(size, device='cuda', ...)
    ↓
at::empty(size, device='cuda')
    ↓
result.normal_(0, 1, generator)
    ↓
...
```

### 4. 表格格式

**对比表**：
```markdown
| 特性 | 实现A | 实现B |
|------|-------|-------|
| 算法 | ... | ... |
| 复杂度 | ... | ... |
```

**API 列表**：
```markdown
| API | 说明 | 参数 | 文档链接 |
|-----|------|------|----------|
| ... | ... | ... | [详见](link) |
```

### 5. 文档语言风格

**原则**：
- ✅ 技术准确性优先
- ✅ 使用中文，专业术语保留英文
- ✅ 代码注释使用中文
- ✅ 先讲概念，再讲实现，最后讲原理
- ❌ 避免口语化表达
- ❌ 避免过度简化导致不准确

**术语规范**：
- RNG: Random Number Generator（随机数生成器）
- PRNG: Pseudo-Random Number Generator（伪随机数生成器）
- CBRNG: Counter-Based RNG（计数器模式 RNG）
- Generator: 生成器（PyTorch 的抽象类）
- Kernel: 核函数（CUDA）
- Thread: 线程
- Block: 块（CUDA）
- Grid: 网格（CUDA）

---

## 文档依赖关系

```
architecture.md (总览)
    ├─→ uniform_call_flow.md (已完成)
    │     └─→ Philox 算法详解
    │
    ├─→ normal_call_flow.md (待写)
    │     └─→ Box-Muller 变换详解
    │
    ├─→ discrete_call_flow.md (待写)
    │     └─→ Fisher-Yates Shuffle
    │
    ├─→ special_distributions.md (待写)
    │     └─→ 逆变换采样原理
    │
    └─→ sampling.md (待写)
          └─→ Alias Method, BTRS 算法
```

**复用内容**：
- Grid-Stride Loop（所有 CUDA 文档复用）
- Generator 机制（所有文档链接到总览）
- 执行策略计算（CUDA 文档复用）

---

## 写作优先级建议

### Phase 1: 核心文档（优先完成）
1. ✅ `architecture.md` - 已完成
2. ✅ `uniform_call_flow.md` - 已完成
3. ✅ `normal_call_flow.md` - 已完成

### Phase 2: 扩展文档
4. 📝 `discrete_call_flow.md`
5. 📝 `special_distributions.md`
6. 📝 `sampling.md`

### Phase 3: 维护和更新
- 根据用户反馈更新
- 添加更多示例
- 补充性能测试数据

---

## 质量检查清单

在提交文档前，请检查：

### 内容完整性
- [ ] 包含完整的调用流程（Python → C++ → CUDA）
- [ ] 标注所有关键文件位置和行号
- [ ] 提供代码示例
- [ ] 包含算法原理解释

### 技术准确性
- [ ] 代码片段已验证（从实际代码复制）
- [ ] 流程图与实际调用一致
- [ ] 算法描述正确
- [ ] 参数说明准确

### 可读性
- [ ] 结构清晰，层次分明
- [ ] 使用表格和列表组织信息
- [ ] 代码有注释说明
- [ ] 有导航链接（章节跳转）

### 格式规范
- [ ] Markdown 格式正确
- [ ] 代码块指定语言（```cpp, ```python）
- [ ] 表格对齐
- [ ] 中英文混排时有空格

---

## 常见问题（写作者 FAQ）

### Q1: 如何确定文档的详细程度？

**A**: 遵循"三层递进"原则：
1. **概念层**: 告诉读者这是什么、为什么要这样
2. **实现层**: 展示关键代码和流程
3. **原理层**: 在 Appendix 中深入讲解数学/算法

核心内容放主体，深入细节放附录。

### Q2: CUDA 和 CPU 实现都要详细讲吗？

**A**: **CUDA 为重点，CPU 简略**。理由：
- CUDA 实现更复杂（Philox, Grid-Stride Loop）
- CUDA 性能更关键（深度学习主要在 GPU）
- CPU 实现相对简单（MT19937 + 标准库）

CPU 部分只需简要说明算法，不必深入代码细节。

### Q3: 如何处理代码变更？

**A**: 使用 **git blame** 和 **相对路径**：
```markdown
文件位置: `aten/src/ATen/native/Distributions.cpp:250`
```
这样即使行号变化，读者也能通过搜索函数名找到。

### Q4: 是否需要性能测试数据？

**A**: **可选但推荐**。如果有：
- 放在独立的 "Performance" 章节
- 说明测试环境（GPU 型号、PyTorch 版本）
- 提供复现代码

### Q5: 如何处理多个版本的 API？

**A**: 列出所有变体，说明区别：
```markdown
### API 变体

1. `torch.rand(size, ...)` - 最常用
2. `torch.rand(size, *, generator=None, ...)` - 带生成器
3. `torch.rand(size, *, out=None, ...)` - 输出参数版本
4. `torch.rand_like(input, ...)` - 按形状生成

**区别**:
- `rand` 创建新张量
- `rand(..., out=tensor)` 复用现有张量
- `rand_like` 自动推断形状
```

---

## 维护计划

### 定期更新（每季度）
- 检查代码位置是否变更
- 更新 PyTorch 版本信息
- 补充新增的 API

### 根据反馈更新
- 用户提出的问题 → 补充到 FAQ
- 发现的错误 → 及时修正
- 需求的示例 → 添加到文档

### 版本管理
每个文档底部标注：
```markdown
**文档版本**: v1.0
**对应 PyTorch 版本**: >= 2.0
**最后更新**: 2024-12-22
```

---

## 附录：已有文档分析

### uniform_call_flow.md 的优点
✅ 结构清晰，层次分明
✅ 代码示例丰富
✅ CUDA 实现讲解详细
✅ Philox 算法原理深入
✅ 流程图直观易懂

### 可改进的地方
- 可以增加性能对比数据
- 可以增加更多使用场景示例
- 可以添加常见错误和调试方法

### 作为模板的价值
其他文档应该复用：
- 整体结构（1-9 章 + Appendix）
- 流程图风格
- 代码注释方式
- 表格格式

---

## 总结

本文档规划采用 **"按分布类型分文档 + 总览文档"** 的方式，具有以下优势：

1. **模块化**: 每个文档独立，易于维护
2. **可扩展**: 新增内容无需重构
3. **读者友好**: 按需查阅，快速定位
4. **写作者友好**: 结构模板清晰，复用度高

**下一步行动**:
1. 编写 `normal_call_flow.md`（复用现有结构）
2. 根据反馈调整文档风格
3. 逐步补充其他分布文档

---

**规划版本**: v1.0
**制定日期**: 2024-12-22
**维护者**: Documentation Team
