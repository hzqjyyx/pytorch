c10/cuda 是一个核心库，包含 CUDA 相关功能。以下是主要内容：

- **核心定位**：提供 CUDA C API 的 C++ 封装，不包含任何 GPU 内核
- **功能范围**：包含编写 CUDA 代码时通常需要的基础功能
- **特殊要求**：代码会被自动转译到 c10/hip（用于 AMD GPU/ROCm 支持）
- **维护规则**：
  - 添加新文件/功能需要更新 `torch/utils/hipify/cuda_to_hip_mappings.py`
  - 新增的 CUDA 函数需要映射到对应的 HIP 函数
  - 添加新目录需要同时更新 c10/cuda/CMakeLists.txt 和 c10/hip/CMakeLists.txt
