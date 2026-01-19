这个文件是 Android Neural Networks API (NNAPI) 的最小化头文件，主要定义了与 NNAPI 交互所需的核心数据结构和枚举类型。

**主要功能：**

- **结果码枚举** (`ResultCode`)：定义 NNAPI 操作的返回状态，包括成功、内存不足、数据错误、设备不可用等

- **操作数类型枚举** (`OperandCode`)：定义神经网络中支持的数据类型，包括浮点数、整数、布尔值、量化张量等多种格式

- **性能偏好枚举** (`PreferenceCode`)：定义执行模式选项，支持低功耗、快速单次答案、持续速度三种模式

- **不透明结构体声明**：定义了 6 个指针类型的结构体（Memory、Model、Device、Compilation、Execution、Event），用于与 NNAPI 库交互

- **操作数类型结构体** (`ANeuralNetworksOperandType`)：定义张量的元数据，包括数据类型、维度信息、量化参数（scale 和 zeroPoint）

- **动态加载支持**：注释说明所有函数通过 dlopen/dlsym 动态加载，操作码从序列化模型中提取，避免直接依赖 NNAPI 库的完整头文件
