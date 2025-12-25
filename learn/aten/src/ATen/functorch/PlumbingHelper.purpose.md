## PlumbingHelper 文件功能分析

这两个文件实现了 functorch 中的 vmap "管道"（plumbing）层，用于连接用户定义的批处理规则和 PyTorch 的操作符调度系统。

### 核心概念

vmap 中的批处理规则（batch rule）与原始操作符签名不同。例如：
- 原始操作符：`at::sin(Tensor self)`
- 批处理规则：`sin_batch_rule(Tensor self, std::optional<int> self_bdim)`

管道层负责将用户定义的批处理规则包装成能注册到 Batched key 的内核形式。

### 主要功能函数

**vmap_check_escaped**
- 验证张量是否在 vmap 函数内部逃逸
- 如果张量逃逸会抛出错误

**makeBatched**
- 根据张量、batch 维度和 level 创建批处理张量
- 支持 `std::optional<int64_t>` bdim，若无值则返回原始张量

**unwrapTensorAtLevel**
- 从可能的 BatchedTensor 中解包张量
- 返回元组：(unwrapped_tensor, bdim)
- 若张量未批处理或 level 不匹配，返回 (tensor, std::nullopt)

**isBatchedAtLevel** （多个重载）
- 检查单个张量、可选张量、张量列表是否在特定 level 被批处理
- 支持多种容器类型：`ITensorListRef`, `c10::List<std::optional<Tensor>>`, `ArrayRef`

**ivalueParticipatesInCurrentLevel**
- 内联函数，检查 IValue 是否在当前 vmap level 参与批处理
- 仅对张量类型进行检查

**makeBatchedVector**
- 批量创建多个批处理张量

---

### 功能概览

- **张量包装/解包**：在批处理和非批处理形式间转换
- **Level 管理**：通过 level 标识不同 vmap 嵌套层
- **逃逸检测**：防止张量意外逃出 vmap 作用域
- **容器适配**：支持多种 PyTorch 容器类型检查
- **调度支持**：为操作符调度系统提供批处理张量识别能力
