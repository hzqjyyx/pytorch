## 主要功能

这是 PyTorch 的 NNAPI (Android Neural Networks API) 动态加载包装器，提供在 Android 设备上运行神经网络推理的能力。

### 核心设计模式

**双包装器架构**：
- `nnapi_`: 直接包装 NNAPI 函数指针，从 `libneuralnetworks.so` 动态加载
- `check_nnapi_`: 包装错误检查版本，每个函数调用前先 `CAFFE_ENFORCE` 验证函数指针存在，调用后检查返回值是否为 `ANEURALNETWORKS_NO_ERROR`

### 关键流程

**动态加载机制** (`nnapi_wrapper_load`)：
1. Windows 平台直接抛出异常（不支持）
2. 使用 `dlopen("libneuralnetworks.so")` 加载 Android NNAPI 库
3. 通过 `dlsym` 查找所有 NNAPI 函数符号（27+ 个函数）
4. 同时初始化原始函数指针和检查包装器
5. 使用静态 `loaded` 标志确保只加载一次

**包装的 API 分类**：

1. **设备查询**:
   - `_getDeviceCount/Device`: 枚举 NPU/加速器设备
   - `Device_getName/getVersion/getFeatureLevel`: 查询设备信息

2. **模型构建** (`Model_*`):
   - `create/free/finish`: 生命周期管理
   - `addOperand/setOperandValue`: 添加张量操作数
   - `addOperation/identifyInputsAndOutputs`: 构建计算图
   - `relaxComputationFloat32toFloat16`: 精度优化选项

3. **编译** (`Compilation_*`):
   - `create/createForDevices`: 为设备编译模型
   - `setPreference`: 设置延迟/功耗/性能偏好
   - `finish`: 完成编译

4. **执行** (`Execution_*`):
   - `create/compute/startCompute`: 同步/异步推理
   - `setInput/setOutput`: 设置输入输出（支持内存或直接缓冲区）
   - `getOutputOperandRank/Dimensions`: 动态形状查询

5. **内存管理**:
   - `Memory_createFromFd`: 从文件描述符创建共享内存
   - `Event_wait/free`: 异步执行同步

### 错误处理策略

所有 `check_*` 包装函数：
- 调用前验证函数指针非空
- 捕获返回值并强制要求 `ANEURALNETWORKS_NO_ERROR`
- 使用 `CAFFE_ENFORCE` 宏在失败时抛出异常（包含函数名和错误码）

### 代码生成特征

文件头注释表明由 `nnapi/codegen.py` 自动生成，确保：
- 函数签名与 NNAPI 规范一致
- 检查包装器与原始函数同步
- 减少手动维护错误

### 使用场景

PyTorch Mobile 在 Android 设备上通过此包装器将模型降到硬件加速器（DSP/NPU/GPU），实现：
- 低延迟推理
- 降低功耗
- 利用设备特定优化（如量化支持）
