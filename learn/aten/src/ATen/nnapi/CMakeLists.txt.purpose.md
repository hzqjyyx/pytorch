- **条件编译支持**：通过 `PYTORCH_NNAPI_STANDALONE` 标志区分两种构建模式
  - 独立构建模式：作为独立项目编译，需要找到 Torch 包
  - 树内构建模式：在 PyTorch 源码树内编译

- **独立构建配置**（当 `PYTORCH_NNAPI_STANDALONE` 为真时）
  - 设置 CMake 最低版本要求（3.5）
  - 定义项目名称为 `pytorch_nnapi`
  - 指定 C++ 标准为 C++14
  - 查找并链接 Torch 库

- **源文件管理**
  - 独立模式：显式列出三个源文件（`nnapi_bind.cpp`、`nnapi_wrapper.cpp`、`nnapi_model_loader.cpp`）
  - 树内模式：通过 `file(GLOB ...)` 自动收集所有 `.cpp` 文件

- **库构建**
  - 独立模式：构建共享库 `pytorch_nnapi`
  - 树内模式：将源文件列表传递给父作用域供上层 CMake 使用
