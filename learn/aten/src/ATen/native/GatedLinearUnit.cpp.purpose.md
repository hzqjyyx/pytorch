## GatedLinearUnit.cpp 核心功能分析

这个文件实现了 **Gated Linear Unit (GLU)** 操作及其微分计算。

### 主要组件

**1. 元函数 (at::meta::glu)**
- 验证输入张量不是 0 维
- 检查指定维度的大小是偶数
- 将输入沿该维度分成两半，输出大小为输入的一半

**2. GLU 前向计算 (glu_out)**
- 调用设备相关的 stub 函数执行实际计算
- 计算公式: `output = input_first_half * sigmoid(input_second_half)`

**3. GLU 反向传播 (glu_backward_cpu_out)**
- 计算梯度输入：
  - 第一部分：`grad_input_a = grad_output * sigmoid(input_b)`
  - 第二部分：`grad_input_b = grad_output * sigmoid(input_b) * (1 - sigmoid(input_b)) * input_a`
- 使用 TensorIterator 融合计算第二部分以提高性能

**4. GLU Jacobian-向量积 (glu_jvp)**
- 计算前向模式自动微分
- 对输入的微小扰动进行传播

**5. GLU 反向 JVP (glu_backward_jvp)**
- 计算反向传播中的高阶导数
- 处理二阶导数计算

### 核心特点

- **bullet-point 总结:**
  - GLU 激活函数的完整实现（前向、反向、自动微分）
  - 使用 TensorIterator 优化张量操作性能
  - 支持任意维度的分割和计算
  - 包含一阶和二阶导数支持
  - 通过 dispatch 机制支持多种后端设备
