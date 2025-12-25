这两个文件实现了 PyTorch 的**推理模式（Inference Mode）**机制，这是一个性能优化和安全控制的核心组件。

## 主要功能

**InferenceMode** 是一个 RAII（资源获取即初始化）风格的线程本地守卫类，用于在执行推理时禁用自动求导（autograd）功能。

### 核心设计

当进入推理模式时，该守卫会：

1. **修改自动求导状态**（InferenceMode.h:59-63）：
   - 禁用梯度计算（`grad_mode = false`）
   - 启用推理模式标志（`inference_mode = true`）
   - 禁用前向梯度模式
   - 禁用多线程

2. **调整分发键集（DispatchKeySet）**（InferenceMode.h:64-73）：
   - 移除 `ADInplaceOrView` 键从 included 集合
   - 添加所有自动求导相关的键到 excluded 集合
   - 这确保在推理时不会调用自动求导内核

3. **析构时恢复**（InferenceMode.h:81-84）：
   - 恢复之前保存的自动求导状态
   - 恢复之前的分发键配置

### 性能优化

`is_enabled()` 方法（InferenceMode.cpp:9-11）直接查询线程本地存储的推理模式标志，避免每次都检查整个 `DispatchKeySet`，因为这是在 `TensorImpl` 构造函数等频繁调用的关键路径上。

### 使用示例

```cpp
{
    c10::InferenceMode guard(true);  // 启用推理模式
    // 此区间内不会记录梯度
}  // 析构时自动恢复原状态
```

---

## Bullet Points

- **RAII 线程本地守卫**：自动管理推理模式的启用/禁用生命周期
- **禁用自动求导**：在推理时完全关闭梯度计算和反向传播
- **分发键管理**：通过调整 DispatchKeySet 控制哪些算子内核被调用
- **性能关键**：`is_enabled()` 使用独立 TLS 加速频繁查询
- **状态恢复**：析构时完整恢复之前的自动求导和分发键状态
- **多线程安全**：通过线程本地存储实现线程隔离
- **不可复制/移动**：删除拷贝和移动构造函数确保 RAII 语义
