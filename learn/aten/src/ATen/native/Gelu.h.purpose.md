- **文件用途**：定义 GELU（高斯误差线性单元）激活函数的类型枚举和转换工具

- **核心组件**：
  - `GeluType` 枚举类：定义两种 GELU 实现方式
    - `None`：标准 GELU
    - `Tanh`：基于双曲正切的近似 GELU
  
- **主要函数**：
  - `get_gelutype_enum()`：将字符串参数（"none" 或 "tanh"）转换为 `GeluType` 枚举值，无效输入时触发断言
  - `gelutype_to_string()`：将 `GeluType` 枚举值反向转换为字符串表示

- **应用场景**：在 PyTorch 神经网络中选择不同的 GELU 激活函数变体，支持用户指定近似算法
