## 核心功能

这两个文件提供了一套**张量参数验证工具集**，主要用于 ATen（PyTorch 的 C++ 张量库）native 函数的输入检查。由于 native 函数默认不做参数检查，这些工具确保操作前张量满足特定条件。

## 主要组件

### 1. 参数包装类型

**TensorArg**
- 包装 `const Tensor&`、名称和位置（1-indexed）
- 禁止右值引用构造，避免悬垂引用
- 提供 `->` 和 `*` 操作符访问底层张量

**TensorGeometryArg**
- 包装 `TensorGeometry`（张量的形状/步长等几何信息，不含数据）
- 可从 `TensorArg` 隐式构造
- 用于只需检查形状而不需要访问数据的场景

**CheckedFrom**
- 类型别名 `const char*`，描述调用检查的函数名
- 用于错误消息中标识检查来源

### 2. 维度检查

**checkDim**
```cpp
checkDim(c, tensor, "input", 1, 3);  // 检查是否为 3 维张量
```
- 验证张量维度数是否等于指定值
- 有两个重载：直接传 Tensor 或传 TensorGeometryArg

**checkDimRange**
```cpp
checkDimRange(c, t, 2, 5);  // 维度必须在 [2, 5) 范围内
```
- 检查维度是否在 `[dim_start, dim_end)` 区间（左闭右开）

**checkSameDim**
- 验证两个张量维度数相同

### 3. 形状与大小检查

**checkSize**
```cpp
checkSize(c, t, {3, 224, 224});           // 检查整体形状
checkSize(c, t, 0, 10);                   // 检查第 0 维大小为 10
```
- 两种重载：检查整体形状或特定维度大小
- `checkSize_symint` 版本支持符号整数（用于动态形状）

**checkSameSize**
- 验证两个张量形状完全相同
- `checkAllSameSize` 检查多个张量形状一致

**checkNumel**
```cpp
checkNumel(c, t, 1000);  // 检查元素总数
```
- 验证张量元素总数
- `checkSameNumel/checkAllSameNumel` 检查多个张量元素数一致

### 4. 内存布局检查

**checkContiguous**
- 验证张量是否连续存储（`is_contiguous()`）
- `checkAllContiguous` 批量检查，自动跳过未定义的张量

### 5. 设备与后端检查

**checkSameGPU**
```cpp
checkSameGPU(c, t1, t2);
```
- 验证两个张量在同一 GPU 设备上
- 特殊处理：如果任一张量在 CPU 上会报错
- `checkAllSameGPU` 批量检查

**checkBackend**
- 检查张量是否使用指定 Backend（如 CPU、CUDA）

**checkDeviceType**
- 检查张量设备类型（DeviceType 比 Backend 更细粒度）

**checkLayout**
- 验证张量布局（如 strided、sparse_coo）

### 6. 类型检查

**checkScalarType**
```cpp
checkScalarType(c, t, ScalarType::Float);
```
- 验证张量数据类型是否为指定类型

**checkScalarTypes**
```cpp
checkScalarTypes(c, t, {ScalarType::Float, ScalarType::Double});
```
- 检查张量类型是否在允许的类型列表中

**checkSameType**
- 验证两个张量类型完全相同（包括设备、数据类型等）
- `checkAllSameType` 批量检查

### 7. 定义性检查

**checkDefined**
```cpp
checkDefined(c, t);  // 确保张量已定义（非 null）
```
- 验证张量已定义
- `checkAllDefined` 批量检查，**不会**跳过未定义张量

### 8. 工具函数

**maybe_data_ptr**
```cpp
void* ptr = maybe_data_ptr(tensor);  // 未定义返回 nullptr
```
- 安全获取数据指针，未定义张量返回 `nullptr`

**check_dim_size**
```cpp
check_dim_size(tensor, 3, 0, 10);  // 检查 3 维且 tensor.size(0) == 10
```
- 同时检查维度数和特定维度大小

### 9. 步长计算（detail 命名空间）

**defaultStrides**
```cpp
defaultStrides({2, 3, 4});  // 返回 {12, 4, 1}
```
- 计算连续张量的默认步长（C-order，行主序）

**computeStride**
```cpp
auto new_stride = computeStride(
    old_shape, old_stride, new_shape
);  // 返回 optional<vector<int64_t>>
```

**核心算法**（`computeStride_impl`）：
1. **分块思想**：将 `oldshape` 分割成"连续块"
   - 连续块条件：`oldstride[i] = oldshape[i+1] * oldstride[i+1]`
   
2. **匹配验证**：`newshape` 必须能分成相同数量的块
   - 每个新块的元素总数（numel）必须与对应旧块匹配

3. **特殊情况处理**：
   - 空张量：返回全 1 步长
   - 零元素张量且形状相同：直接复制旧步长
   - 零元素张量但形状不同：计算默认步长

4. **步长计算逻辑**：
   ```cpp
   // 从后向前遍历
   chunk_base_stride = oldstride.back();
   // 累积 view_numel 直到匹配 tensor_numel
   newstride[view_d] = view_numel * chunk_base_stride;
   ```

5. **返回**：
   - 成功：`std::optional` 包含新步长向量
   - 失败（无法匹配）：`std::nullopt`

**三个重载**：
- `IntArrayRef` → `std::vector<int64_t>`
- `SymIntArrayRef` → `SymDimVector`（支持符号形状）
- `IntArrayRef + DimVector` → `DimVector`

## 设计模式

### checkAllSame 模板
```cpp
static void checkAllSame(
    CheckedFrom c, 
    ArrayRef<TensorArg> tensors,
    void(*fn)(CheckedFrom, const TensorArg&, const TensorArg&)
) {
    // 找到第一个定义的张量作为基准
    // 用函数指针 fn 对后续每个张量进行两两比较
}
```
- 统一实现 `checkAllSameSize`、`checkAllSameNumel`、`checkAllSameType` 等
- 自动跳过未定义张量

### 错误消息格式
所有检查失败时输出统一格式：
```
Expected <condition>, but got <actual> for <tensor_arg> (while checking arguments for <CheckedFrom>)
```

例如：
```
Expected 3-dimensional tensor, but got 2-dimensional tensor for argument #1 'input' (while checking arguments for conv2d)
```

## 典型使用场景

```cpp
// Native 函数示例
Tensor my_conv2d(const Tensor& input, const Tensor& weight) {
    checkDim(__func__, input, "input", 1, 4);      // 4D 输入
    checkDim(__func__, weight, "weight", 2, 4);    // 4D 卷积核
    checkSameGPU(__func__, {input, 1}, {weight, 2});
    checkContiguous(__func__, {input, 1});
    
    // 实际计算...
}
```

---

## 其他相关内容（简要）

- **ROCm 相关**：`checkSameGPU` 等函数同样适用于 AMD ROCm 设备检查
- **Backward 相关**：这些检查常用于前向和反向传播的输入验证，确保梯度计算前张量合法
