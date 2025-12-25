**CompileTimeFunctionPointer.h 的主要功能**

这个文件实现了一个类型系统，用于在编译时捕获函数指针，使得编译器能够：

- **编译时函数指针封装**：通过模板 `CompileTimeFunctionPointer<FuncType_, func_ptr_>` 将函数指针作为类型参数，让编译器在编译期就知道具体是哪个函数

- **内联优化**：由于函数指针在编译时已知，编译器可以将函数调用内联，避免函数指针间接调用的开销

- **两种使用方式**：
  - `TORCH_FN_TYPE(func)`：返回类型，可用于模板参数或类型别名
  - `TORCH_FN(func)`：返回实例，可直接传递给函数模板

- **类型检查**：通过 `is_compile_time_function_pointer` trait 可以检测一个类型是否为编译时函数指针

- **应用场景**：用于 PyTorch 框架中需要高性能函数指针调用的场景，例如算子分发、回调机制等，通过模板实例化让编译器掌握完整信息进行优化
