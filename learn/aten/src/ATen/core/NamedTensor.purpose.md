## NamedTensor 核心功能

**NamedTensor** 是 PyTorch 中用于支持带名称维度的张量系统。它允许用户给张量的每个维度赋予语义化的名称，而不仅仅使用位置索引。

### 主要组件

**NamedTensorMeta 类** (NamedTensor.h:23-69)
- 存储张量维度名称的元数据容器
- 核心不变量：至少有一个非通配符维度名称
- 提供克隆、查询和更新名称的接口

**NamesMode 机制** (NamedTensor.h:73-76, NamedTensor.cpp:8-17)
- 线程本地全局开关，控制命名张量功能的启用/禁用
- 禁用时，所有张量被视为无名称，通过 DispatchKey::Named 控制调度

**NoNamesGuard RAII 守卫** (NamedTensor.h:81-101)
- 线程本地作用域守卫，临时禁用名称模式
- 析构时自动恢复之前的状态

### 核心操作

**名称设置与验证**
- `internal_set_names_inplace()`: 设置张量维度名称，支持 DimnameList 和移动语义
- `check_names_valid_for()`: 验证名称有效性（唯一性、维数匹配、设备支持）
- `check_unique_names()`: O(N²) 检查确保无重复名称

**名称查询**
- `get_opt_names()`: 返回张量已分配的名称（可能为空）
- `get_names()`: 返回张量名称，未命名张量返回全通配符列表
- `has_names()`: 判断张量是否有非通配符名称

### 设计特点

- **库分离限制**：TensorImpl 在 c10 中，Dimname 在 ATen 中，NamedTensor 作为中间层桥接
- **性能考虑**：仅在需要时分配元数据，全通配符名称不占用内存
- **设备支持**：CPU、CUDA、XPU、PrivateUseOne

### 关键实现细节

- 线程本地状态管理，避免全局变量竞争
- 延迟验证支持（validate_names 参数）
- 通过 TensorImpl::named_tensor_meta() 存储元数据指针
- Strided 布局限制

---

**核心职责总结：**

- 为张量维度提供命名和跟踪机制
- 实现全局命名模式开关和线程本地守卫
- 验证维度名称的有效性和唯一性
- 提供元数据的分配、查询和更新接口
- 桥接 c10 和 ATen 库的分离限制
