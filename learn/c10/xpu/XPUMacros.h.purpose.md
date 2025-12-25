## C10_XPU_EXPORT 和 C10_XPU_IMPORT 宏定义

这个文件为 XPU（Intel GPU）库定义了符号导出/导入宏，用于控制库中函数和变量的可见性。

**跨平台处理：**

- **Windows (_WIN32)：** 使用 `__declspec(dllexport/dllimport)` 控制 DLL 符号导出
- **非 Windows 系统：** 使用 GCC 的 `__attribute__((__visibility__))`，或不做特殊处理

**C10_XPU_API 宏：**

- 在构建主库时 (`C10_XPU_BUILD_MAIN_LIB`) 定义为 `C10_XPU_EXPORT`
- 否则定义为 `C10_XPU_IMPORT`
- 用于标注 libc10_xpu.so 中的公开 API

**核心功能：**

- 定义共享库符号可见性
- 支持条件编译选项 (`C10_USING_CUSTOM_GENERATED_MACROS`)
- 通过 CMake 生成平台特定的宏定义
