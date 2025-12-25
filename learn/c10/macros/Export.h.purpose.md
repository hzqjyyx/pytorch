## C10_MACROS_EXPORT_H 文件功能

这个文件定义了 PyTorch C10 库的符号导出/导入机制，用于处理跨平台的动态库可见性问题。

**核心机制：**

在不同平台使用不同的编译器属性来控制符号可见性：
- **Windows**: 使用 `__declspec(dllexport)` / `__declspec(dllimport)`
- **Linux/Unix (GCC)**: 使用 `__attribute__((__visibility__("default")))` 和 `hidden`
- **其他编译器**: 宏展开为空（无效果）

**关键宏定义：**

1. **C10_EXPORT / C10_IMPORT** - 基础导出/导入宏，根据 `C10_BUILD_SHARED_LIBS` 在编译时决定行为

2. **C10_API** - libc10.so 的公共符号标记
   - 定义 `C10_BUILD_MAIN_LIB` 时 = C10_EXPORT
   - 否则 = C10_IMPORT

3. **TORCH_API** - libtorch.so 的公共符号标记
   - 定义 `CAFFE2_BUILD_MAIN_LIB` 时 = C10_EXPORT
   - 否则 = C10_IMPORT

4. **TORCH_CUDA_CU_API / TORCH_CUDA_CPP_API** - CUDA 相关库的导出宏（处理大型二进制链接问题）

5. **TORCH_HIP_API / TORCH_XPU_API** - AMD GPU 和 Intel GPU 相关导出宏

**使用规则：**

- 建议不要混合静态/动态库构建（如 c10 是动态库，依赖项也应该是动态库）
- 支持自定义宏生成（通过 `C10_USING_CUSTOM_GENERATED_MACROS` 标志）
- 在库源文件中用对应 API 宏注解公共符号

**输出总结：**

- ✓ 跨平台编译器抽象层
- ✓ 动态库符号可见性控制
- ✓ 多个库导出宏（C10、Torch、CUDA、HIP、XPU）
- ✓ 静态/动态库构建时的条件编译
