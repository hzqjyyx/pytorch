## 文件功能分析

这个文件实现了量化线性层权重的解包（unpack）操作，支持多个量化后端。

**核心机制：**
- 通过运行时多态性，利用 `packed_weight` 指针（类型为 `intrusive_ptr<LinearPackedParamsBase>`）在运行时指向不同的实现类（PackedLinearWeightsQnnp、PackedLinearWeights 或 PackedLinearWeightsCudnn）
- 具体的解包实现分别位于 `/cpu/LinearUnpackImpl.cpp`（fbgemm & qnnpack）和 `/cudnn/linear_unpack_impl.cpp`（cudnn）

**主要类和功能：**

- **QLinearUnpackWeightInt8**：解包 Int8 量化的权重，直接调用 `packed_weight->unpack()`

- **QLinearUnpackWeightFp16**：解包 Fp16 量化的权重，额外检查后端不能是 QNNPACK（不支持）

- **QLinearUnpackWeightInt8Legacy** & **QLinearUnpackWeightFp16Legacy**：处理旧版本接口，强制用户升级到新的 `LinearPackedParamsBase` 重载

**库注册：**

- CPU 后端：注册旧版接口（legacy），返回错误提示
- CatchAll 后端：注册 `LinearPackedParamsBase` 类、新版 Int8 和 Fp16 解包操作

**关键点：**

- 提供统一的解包接口，支持 fbgemm、qnnpack 和 cudnn 三个量化后端
- 通过多态设计解耦不同后端的具体实现
- 向后兼容性：旧接口已弃用，引导用户迁移到新 API
