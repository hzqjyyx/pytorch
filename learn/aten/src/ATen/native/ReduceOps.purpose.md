## ReduceOps.h (头文件)

**定义调度桩和函数签名，用于各种归约操作的多后端分发。**

### 核心内容：

1. **Dispatch 函数指针类型定义**
   - `reduce_fn`: 标准归约函数 `void(*)(TensorIterator &)`
   - `reduce_std_var_function`: 标准差/方差 `void(*)(TensorIterator&, correction, take_sqrt)`
   - `reduce_norm_fn`: 范数归约 `void(*)(Tensor&, const Tensor&, Scalar, optional<int64_t>)`
   - `structured_cum_fn` / `cum_fn`: 累积操作函数

2. **声明的 Dispatch Stubs** (通过 `DECLARE_DISPATCH` 宏)
   - 基础归约：`sum_stub`, `nansum_stub`, `prod_stub`, `mean_stub`
   - 逻辑归约：`and_stub`, `or_stub`
   - 极值：`min_values_stub`, `max_values_stub`, `argmax_stub`, `argmin_stub`
   - 统计：`std_var_stub`, `norm_stub`
   - 累积：`cumsum_stub`, `cumprod_stub`, `logcumsumexp_stub`
   - 其他：`aminmax_stub`, `aminmax_allreduce_stub`

3. **公开 API**
   - `var_mean_out`: 用于 Normalization.cu 的方差均值计算

---

## ReduceOps.cpp (实现文件)

**实现所有归约操作的核心逻辑，包括元数据推导、内核调度和特殊情况处理。**

### 主要功能模块：

#### 1. **Meta Functions** (元数据推导，行 134-401)
   - 推导输出张量的形状、dtype
   - 参数验证（维度检查、空张量处理）
   - 实现函数：`all/any`, `argmax/argmin`, `cumsum/cumprod`, `sum/prod/mean`, `norm`, `aminmax`, `amax/amin`

#### 2. **基础归约操作**
   - **Sum** (行 1205-1242): 支持低精度累加缓冲区优化 (`should_use_acc_buffer`)
   - **Nansum** (行 1244-1270): 忽略 NaN 的求和，整数类型直接调用 sum
   - **Prod** (行 1316-1354): 累积乘法
   - **Mean** (行 1356-1436): CPU 上 BF16/FP16 通过 FP32 中间结果提高精度
   - **Nanmean** (行 1438-1470): 计算非 NaN 元素数量后除以 nansum

#### 3. **累积操作**
   - **Cumsum/Cumprod** (行 491-505): 沿维度累积求和/乘积
   - **Cummax/Cummin** (行 788-864): 返回累积极值和索引
   - **Logcumsumexp** (行 445-472): 数值稳定的 log-sum-exp 累积
   - **Cumprod Backward** (行 511-750): 
     - 支持复数（通过共轭）
     - 零值特殊处理（分 k<z1, k=z1, k>z1 三种情况）
     - O(n²) 慢路径用于二阶梯度

#### 4. **逻辑归约**
   - **All/Any** (行 1594-1716): 
     - 支持 uint8 兼容性
     - CUDA 动态类型转换优化
     - 提供 dims 批量处理的默认实现

#### 5. **极值操作**
   - **Amax/Amin** (行 1718-1732): 调用 `max_values_stub` / `min_values_stub`
   - **Aminmax** (行 408-427): 同时计算最小和最大值
   - **Argmax/Argmin** (行 1734-1783): 返回极值索引，支持无 dim 的展平处理

#### 6. **统计操作**
   - **Std/Var** (行 1844-2157):
     - 复数类型：分别计算实部/虚部方差后相加
     - CPU 全归约优化路径 (`std_var_all_cpu`)
     - 支持 unbiased/correction 参数
   - **Std_mean/Var_mean**: 同时返回标准差/方差和均值

#### 7. **范数操作**
   - **Norm** (行 1537-1592): 委托给 `linalg_vector_norm`
   - **Logsumexp** (行 1472-1535): 数值稳定实现 `log(sum(exp(x)))`
   - **Dist** (行 2232-2234): 两张量距离 `norm(self - other, p)`

#### 8. **差分/梯度**
   - **Diff** (行 880-1009): n 阶差分，支持 prepend/append
   - **Gradient** (行 1011-1181): 数值梯度（中心差分/边缘差分）
   - **Cummaxmin Backward** (行 866-878): 使用 scatter_add

#### 9. **稀疏张量支持** (行 2332-2383)
   - COO/CSR 格式的 sum 实现

#### 10. **辅助功能**
   - **Trace** (行 1286-1314): 对角线元素求和
   - **Equal** (行 2236-2308): 张量相等性比较（NaN 敏感）
   - **Value-selecting reduction backward** (行 2314-2330): 用于 max/min/topk 的反向传播

---

### 关键设计模式：

1. **TensorIterator 驱动**: 大多数归约通过 `TensorIterator` 构建计算图
2. **Stub 分发机制**: CPU/CUDA/XPU 后端通过 `DEFINE_DISPATCH` + `stub(device_type, ...)` 调用
3. **精度提升策略**: 
   - 整数 → Long
   - BF16/FP16 → Float (CPU mean/sum)
4. **复数处理**: 分离实部/虚部独立计算
5. **Named Tensor**: 通过 `NoNamesGuard` 和 `propagate_names` 处理

---

### ROCm/Backward 相关（简要）
- **ROCm**: 无显式 ROCm 代码，通过 dispatch 机制间接支持
- **Backward**:
  - `cumprod_backward` (行 511-750)
  - `cummaxmin_backward` (行 866-878)
  - `value_selecting_reduction_backward` (行 2314-2330)
