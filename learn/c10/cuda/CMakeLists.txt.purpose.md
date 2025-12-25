这个CMakeLists.txt文件是PyTorch中C10 CUDA库的构建配置文件。主要功能包括：

- **包含公共构建工具**：引入cmake工具和CUDA相关配置
- **生成配置头文件**：将cuda_cmake_macros.h.in模板配置为cuda_cmake_macros.h，用于编译时宏定义
- **定义源文件列表**：收集C10_CUDA_SRCS中的9个.cpp源文件（分配器、函数、流、守卫等）
- **定义头文件列表**：收集C10_CUDA_HEADERS中的11个.h公共头文件
- **构建C10 CUDA库**：调用torch_cuda_based_add_library()编译成c10_cuda库
- **配置编译选项**：设置dllimport/dllexport宏、隐藏符号可见性、驱动API支持等
- **链接依赖**：链接c10库和CUDA运行时库(torch::cudart)，非Windows系统还链接dl库
- **设置包含路径**：配置编译和安装时的头文件搜索路径
- **安装目标**：将库文件安装到lib目录，头文件安装到include/c10/cuda目录
- **构建测试**：通过add_subdirectory(test)添加测试子目录
- **支持分布式构建**：通过BUILD_LIBTORCHLESS选项支持不同的构建模式
