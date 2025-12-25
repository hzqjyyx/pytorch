# TunableOp 主要功能理解

## 核心目的
为多实现方式的算子（如 GEMM）自动选择最快的实现。某些操作可能有多个库或技术实现，TunableOp 通过基准测试找出最优方案。

## 工作机制

**两级开关设计：**
- 启用 TunableOp：用 Tunable 实现替换标准算子
- 启用 Tuning：控制是否进行性能调优

**执行流程：**
1. 调用 TunableOp 时先查询该输入是否已调优
2. 如已调优 → 直接使用已知最快实现
3. 如未调优且启用 tuning → 对所有注册实现进行基准测试，选择最快的
4. 如未调优且未启用 tuning → 使用默认实现

## 文件存储系统

**输入输出机制：**
- 默认文件名：`tunableop_results.csv`
- 多 GPU 时自动插入设备序号避免冲突
- 启动时读取已有调优结果
- 退出时写入包含新旧所有调优结果

**CSV 格式示例：**
```
Validator,PT_VERSION,2.2.0
Validator,ROCM_VERSION,6.0.0.0-12969-1544e39
GemmTunableOp_float_NT,nt_25088_4096_64,1219,1.262
GemmTunableOp_float_NT,nt_4096_4096_64,1216,0.033
```

**四字段含义：**
1. 算子名称
2. 算子参数（如 GEMM 的 M,N,K 维度）
3. 解决方案名称（可手动编辑为 "Default" 或特定索引）
4. 平均执行时间（可选）

**版本验证：**
- Validator 行记录 PyTorch、库版本信息
- 版本变化时拒绝旧调优文件（因性能特征可能改变）

## 调优行为

**性能测试方法：**
- 遍历所有注册实现，逐个分析性能
- 默认每个方案运行 100 次或 30ms 内尽可能多次（取较小值）
- 计算平均执行时间，选择最快且准确的方案
- 失败条件：精度不达标或返回错误码

**缓存影响处理：**
- 可选 warmup 阶段帮助硬件达到稳定功耗状态
- 提供指令缓存刷新选项
- 提供输入张量轮换机制
- 目的：让测试环境更接近实际工作负载

## 离线调优

**适用场景：**
- 高内存占用工作负载（在线调优可能 OOM）
- 计算密集型任务（收集一次 GEMM，用不同参数反复调优更高效）

**单 GPU 工作流：**
```python
# 步骤 1: 收集未调优的 GEMM
PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_RECORD_UNTUNED=1

# 步骤 2: 离线调优
import torch.cuda.tunable as tunable
tunable.tune_gemm_in_file("tunableop_untuned0.csv")
```

**多 GPU 分布式调优：**
```python
if __name__ == "__main__":
    num_gpus = 8
    tunable.mgpu_tune_gemm_in_file("tunableop_untuned?.csv", num_gpus)
```

处理流程：
1. 收集多个文件中的 GEMM，去重
2. 分配到多个 GPU 并行调优
3. 汇总到 `tunableop_results_full0.csv`
4. 为每个 GPU 复制一份结果文件

## 控制接口

**三种方式（优先级递减）：**
1. 环境变量（首次读取后固定，不可编程修改）
2. C++ API：`at::cuda::tunable::getTuningContext()`
3. Python API：`torch.cuda.tunable` 模块

**关键参数：**
- `ENABLED`/`enable()`: 启用 TunableOp
- `TUNING`/`tuning_enable()`: 启用调优（默认开启）
- `RECORD_UNTUNED`: 记录未调优算子用于离线调优
- `MAX_TUNING_DURATION_MS`: 单方案最大测试时间（默认 30ms）
- `MAX_TUNING_ITERATIONS`: 单方案最大迭代次数（默认 100）
- `NUMERICAL_CHECK`: 数值正确性检查
- `VERBOSE`: 调试日志级别（0-3）

---

**ROCm 相关：**
- 当前主要实现为 ROCm 的 TunableGemm
- 可选择 rocblas 或 hipblaslt 库的不同算法
- CUDA 构建可用但只有默认实现

**其他特性：**
- warmup 迭代次数/时长配置
- 旋转缓冲区大小控制（用于模拟冷缓存）
- icache 刷新开关
- BLAS 参数日志记录
