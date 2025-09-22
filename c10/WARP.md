# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Overview

C10 is PyTorch's core tensor library providing fundamental abstractions for tensors, devices, dispatch, and memory management. It is designed as a minimal dependency foundation that other PyTorch components build upon. The library maintains strict requirements for minimal dependencies and should never depend on implementation-specific or backend-specific libraries.

## Build System & Common Commands

### Building
The project uses CMake with a minimum requirement of version 3.18. The main build configuration is in the root `CMakeLists.txt`.

```bash
# Configure and build (typically from PyTorch root directory)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Running Tests
Tests are located in the `test/` directory and are built automatically when `BUILD_TEST` is enabled.

```bash
# Run all C10 tests
cd build && ctest -R "c10_.*"

# Run a specific test
./build/bin/c10_Device_test
```

Individual test executables follow the pattern `c10_<test_name>` and can be found in the build directory.

### Static Analysis
The project supports include-what-you-use (IWYU) for header cleanup:

```bash
cmake .. -DC10_USE_IWYU=ON
```

## Architecture Overview

### Core Dispatch System
C10 implements PyTorch's multiple dispatch mechanism through several key components:

- **DispatchKey**: Identifies specific "levels" in the dispatch hierarchy (backends, functionalities)
- **DispatchKeySet**: A 64-bit bitset representing active dispatch keys for a tensor
- **BackendComponent**: Represents different compute backends (CPU, CUDA, XLA, etc.)
- **Functionality Keys**: Per-backend customizable features (Dense, Sparse, Quantized, Autograd)

The dispatch system uses a carefully designed bitset representation that balances functionality and performance:
- Backends × Functionalities creates a cross-product of possible dispatch targets
- Some keys are "building blocks" that combine to form runtime keys
- The system is limited to ≤16 backends and ≤64 total entries to fit in efficient bitsets

### Device and Backend Management
The library supports multiple backend types including:
- Standard backends: CPU, CUDA, HIP, XLA, MPS, IPU, XPU, HPU, VE, MTIA
- Private use backends: PrivateUse1, PrivateUse2, PrivateUse3
- Special backends: Meta (no-storage tensors), Lazy

**Important**: Code in `cuda/` gets automatically transpiled to `hip/` for AMD GPU builds. When adding new functionality to the CUDA directory, you must update the mappings in `torch/utils/hipify/cuda_to_hip_mappings.py`.

### Tensor Implementation (TensorImpl)
`TensorImpl` is the core tensor representation providing:
- Storage management and memory layout
- Size, stride, and shape information  
- Device and dtype tracking
- Dispatch key management
- Autograd integration
- Thread-safe copy-on-write storage support

The copy-on-write system in `core/impl/` enables lazy tensor copies while maintaining PyTorch's aliasing invariants. It uses reader-writer locks to handle concurrent materialization safely.

### Memory and Storage
- `Storage` provides the underlying data allocation abstraction
- `Allocator` interface supports custom memory allocation strategies
- Device-specific allocators (CPU, CUDA) handle backend memory management
- Copy-on-write storage enables efficient lazy copying with proper thread safety

### Utility Components
- `util/`: General C++ utilities and polyfills not specific to deep learning
- `core/impl/`: Internal implementations with no backward compatibility guarantees
- `macros/`: Build configuration and compiler-specific macros
- `mobile/`: Mobile-specific optimizations and implementations

## Key Development Guidelines

### Dependencies
C10 must maintain minimal dependencies. Any new dependency affects all PyTorch components transitively. Check with core developers before adding dependencies.

### Thread Safety
The copy-on-write storage system requires careful attention to thread safety:
- Lazy-clone operations are treated as reads
- Materialization (writing to tensors) must be properly synchronized
- The system uses reader-writer locks for concurrent materialization scenarios

### Backend Extensions
When adding new backends:
1. Update `BackendComponent` enum in `DispatchKey.h`
2. Add corresponding entries to all relevant dispatch key ranges
3. Update autograd and autocast mappings in `DispatchKeySet.h`
4. For CUDA code, add HIPIFY mappings for AMD GPU compatibility

### Testing
- All new functionality requires comprehensive unit tests
- Tests should be placed in `test/` with the naming pattern `*_test.cpp`
- Tests automatically build when `BUILD_TEST` is enabled
- Use GoogleTest framework for test implementation

### CUDA/HIP Compatibility
Code in `cuda/` is automatically transpiled to `hip/` for AMD GPU support:
- Use generic CUDA programming patterns when possible
- Update `C10_MAPPINGS` in the HIPIFY configuration for new symbols
- Test both CUDA and HIP builds when making changes

### Performance Considerations
- C10 is a performance-critical foundation library
- Virtual function calls are minimized where possible (e.g., TensorImpl methods)
- The dispatch system is optimized for runtime performance
- Memory allocations and copies are carefully managed
