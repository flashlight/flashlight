# The `fl::Tensor` Framework

**Note: `fl::Tensor` is a work in progress.** Potential API breakage is possible as development continues. We'll do our best to adapt existing code as the API evolves.

`fl::Tensor` is an effort to create a portable tensor frontend that facilitates integrating custom tensor computation stacks for research at the *library, compiler, and hardware level*. It provides:
- An **unopinionated** API with which tensor libraries can be implemented or wrapped
- A **small implementation surface** with few required operators;
- Compute-level functions that help define implementation requirements;
- Tests to ensure equivalent expected behavior;
- Benchmarks with Flashlight to test end-to-end performance with a variety of state-of-the-art neural networks.

In general, the framework *eschews opinionation* and provides the minimum viable surface on top of which to integrate a new tensor backend with *minimal footprint*, requiring the user define only a few dozen operations. **`fl::Tensor` is not an IR**: the user is to define everything under the base interface.

### A Small, Unopinionated API Surface

Tensor stack research can be challenging in that large-scale research frameworks can be opinionated, providing rigid APIs or IRs on top of which to develop compilers and require significant changes to modify internals. `fl::Tensor` prioritizes the following to facilitate development:
1. **Small API footprint**: a small, simple interface to add tensor ops.
2. **User-defined internals**: implement things under a standard, high-level API; no pre-specified IRs.
3. **Compute-model flexibility**: users can integrate JITs, eager-evaluation libraries, and others, with few requirements.

## Implementing or Adding a Tensor Backend
At a high level, implementing an `fl::Tensor` backend is simple and requires subclassing two interfaces:
1. [`TensorAdapterBase`](https://github.com/flashlight/flashlight/blob/master/flashlight/fl/tensor/TensorAdapter.h), which defines operations for member functions on an `fl::Tensor`.
2. [`TensorBackend`](https://github.com/flashlight/flashlight/blob/master/flashlight/fl/tensor/TensorBackend.h) which defines global tensor operations.

Tensor backend implementations should be placed in `tensor/backend/[backend name]`; for example, the ArrayFire backend implementation resides in `tensor/backend/af`. [`ArrayFireTensor.h`](flashlight/fl/tensor/backend/af/ArrayFireTensor.h) and [`ArrayFireBackend.h`](flashlight/fl/tensor/backend/af/ArrayFireBackend.h) provide good starting point examples of implementations.

### Example with an Implementation Stub
In this example, we'll integrate a new `FooBaz` backend.

First, we'll add derived implementations of the `TensorAdapterBase` and `TensorBackend` interfaces. We've created a new directory: `flashlight/fl/tensor/backend/foobaz` and added a `CMakeLists.txt` that links dependency and compiles files with our implementations:
```cmake
cmake_minimum_required(VERSION 3.10)

find_package(FooBazBackendDep REQUIRED) # if our backend depends on something external
target_link_libraries(flashlight PUBLIC FooBazBackendDep::FooBazBackendDep)

target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/FooBazTensor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/FooBazBackend.cpp
)

# If a backend requires a a compute runtime, e.g. CUDA, this can be set
# globally to ensure CUDA interoperability with other FL components:
fl_set_backend_state(ENABLE CUDA)
```
and we'll gate building our backend in `flashlight/fl/tensor/CMakeLists.txt`:
```cmake
# ...
# adding a CMake option to build the backend. Passing -DFL_USE_FOOBAZ=ON when
# running cmake will build the backend.
option(FL_USE_FOOBAZ "Build FooBaz tensor backend" OFF)

if (FL_USE_FOOBAZ)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/foobaz/CMakeLists.txt)
endif()

# ...

# Add a compile definition for compile-time use
target_compile_definitions(
  flashlight
  PUBLIC
  FL_USE_FOOBAZ=$<BOOL:${FL_USE_FOOBAZ}> # added
)
# ...
```

Now, we'll add `FooBazTensor.h`, also in `flashlight/fl/tensor/backend/foobaz` which defines the class `FooBazTensor`, derived from [`TensorAdapterBase`](./TensorAdapterBase.h), as follows:
```cpp
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Index.h"

class FooBazTensor : public fl::TensorAdapterBase {
  // We can store state and metadata associated with our tensor here.
 public:
  // Provide a constructor to create a FooBazTensor using some existing data/a buffer:
  FooBazTensor(
    const Shape& shape,
    fl::dtype type,
    void* ptr,
    Location memoryLocation);
  // ...

  // Add the backend type to the TensorBackendType enum in TensorBase.h, then implement:
  TensorBackendType backendType() const override;

  // Override other interface functions. For example:
  const Shape& shape() override; // returns a Flashlight shape
  void device(void** out) override; // returns a pointer to device memory, if applicable.
  Tensor index(const std::vector<Index>& indices) override; // indexing operator
  // ...
};
```
We write our implementations in `FooBazTensor.cpp` accordingly (omitted here). If our `FooBazTensor` doesn't support some operations, we can have them throw exceptions, or alternatively include, depend on, and delegate to implementations in another existing `fl::Tensor` backend. See the [tensor adapter interface documentation](./TensorAdapter.h) for function-level expected behavior.

Now, we'll create our backend implementation, `FooBazBackend.h`, also in `flashlight/fl/tensor/backend/foobaz`, where we'll define most tensor operations and can store global state related to our backend:
```cpp
#include "flashlight/fl/tensor/TensorBackend.h"

class FooBazBackend : public fl::TensorBackend {
  // We can store global state for our backend (dataflow graphs,
  // memory management, compute streams, op status, etc) here.
 public:
  // Performs pre-flight tasks when initializing the backend
  FooBazBackend();
  // ...

  // Ops on tensors. For example:
  Tensor transpose(const Tensor& tensor, const Shape& dims /* = {} */) override;
  Tensor exp(const Tensor& tensor) override;
  Tensor matmul(const Tensor& lhs, const Tensor& rhs) override;
  // ...
};
```
Implementations of this interface are in `FooBazBackend.cpp` (or distributed across other compilation units), and are ommitted here. As with `FooBazTensor`, if our backend doesn't support some operations, we can have those operations throw exceptions, or alternatively delegate to another existing backend. See the [backend interface documentation](./TensorBackend.h) for function-level expected behavior.

### Testing an Implementation Against Requirements
We can test an `fl::Tensor` implementation against requirements for expected behavior by running test in `flashlight/fl/test/tensor`. These include:
- `TensorBaseTest`, which tests most interface and compute operations
- `IndexTest`, which tests tensor indexing and some in-place operations
- `ComputeTest`, which tests async computation and data availability, as applicable.

## Implementation Requirements
Below is a more formal definition of the implementation requirements for a tensor backend.

In addition to deriving from the `TensorAdapterBase` and `TensorBackend` interfaces, implementers are required to:
- **Provide memory interoperability** as defined in interfaces. This includes constructors to create tensors from arbitrary buffers either on the host or accelerator devices. These include:
  - `void fl::TensorAdapter::host(void** out)`, which provides a host-side host for an underlying tensor.
  - `void fl::TensorAdapter::device(void** out)`, which provides a device-side buffer pointing to an underlying tensor. This buffer may not be up-to-date; non-blocking relative synchronization should be used as needed, i.e. through `Tensor::stream()::relativeSync(Stream&)`.
    - To avoid needing to copy device-side buffers, the API also provides `void unlock()`, which signals that a pointer to device memory can be freed if its corresponding tensor(s) are destroyed.
    - Only memory from buffers returned via `Tensor::host` is guaranteed to be up-to-date. Memory returned from `Tensor::device` may still require computation to be up-to-date; relative synchronization should be used in these cases.

- **Implement compute synchronization primitives** as defined by interfaces. These include:
  - `void fl::TensorBackend::sync()`: blocks the calling thread until all computation on all streams associated with active devices is complete.
  - `void fl::TensorBackend::eval(Tensor&)`: launches any kernels that need to be executed to make the tensor's updated value available.
  - *These can be no-ops if the backend computation model so dictates.*
- **Pass all tests** as provided in `flashlight/fl/test/tensor` for implemented operators.

### Computation Model

Backends implemented with `fl::Tensor` can have both lazy and eager evaluation semantics. The `sync` and `eval` requirements above imply support for lazy evaluation, but one or both can be noops as long as implementation requirements are met.

There are no additional implementation requirements around memory usage besides the constructor, `host` and `device` functions above. Only tensors on which these memory-access functions are called need to have explicit buffers.

### Changing Default Behavior
To adjust the default backend with which Flashlight tensors are created, change the types and preprocessor calues in [`TensorAdapter.cpp`](https://github.com/flashlight/flashlight/blob/master/flashlight/fl/tensor/TensorAdapter.cpp) accordingly.
- **Note:** Flashlight's default tensor backend continues to be [ArrayFire](https://github.com/arrayfire/arrayfire), and newly-created tensors will be `fl::Tensor` will call into ArrayFire by default.

## Tensor Interface Extensions

Some operations fall outside of the minimal tensor interface that `fl::Tensor` provides but are still important in machine learning computation. For these operations, Flashlight defines various `TensorExtension`s which allow implementation of these operations and connection to a `TensorBackend` for seamless dispatch.

Tensor extensions derive from `TensorExtension<T>` and can define their own APIs for backends to implement operations. Extensions are registered to a given backend using the `FL_REGISTER_TENSOR_EXTENSION(...)` macro, which takes a tensor extension type and a tensor backend type. *Only one implementation of a given tensor extension can be registered to a given backend*.

For example, to register an extension implementation (in this case registering a `CudnnAutogradExtension` to the `ArrayFire` implementation of `TensorBackend`), we first check that the backends are available and selected to be built by the user, before registering the extension:
```cpp
#if FL_USE_CUDNN
  #if FL_USE_ARRAYFIRE && FL_ARRAYFIRE_USE_CUDA
FL_REGISTER_TENSOR_EXTENSION(
    CudnnAutogradExtension,
    TensorBackendType::ArrayFire);
  #endif // FL_USE_ARRAYFIRE && FL_ARRAYFIRE_USE_CUDA
#endif // FL_USE_CUDNN
```

### A Tensor Extension for Autograd Operations: `AutogradExtension`

[Flashlight's `AutogradExtension`](flashlight/fl/autograd/tensor/AutogradExtension.h) provides an API for implementing operations that frequently have autograd-aware implementations. These include:
- Convolution (`conv2d`)
- Pooling (`pool2d`)
- Batch normalization (`batchnorm`)
- Recurrent neural networks (`rnn`)

Each of the above interface functions have one or more corresponding backward functions (e.g. `pool2dBackward`) with which to compute gradients with respect to their forward parameters.

For documentation on the behavior and input parameters of functions on `AutogradExtension`, see the [documentation for `AutogradOps.h`](flashlight/fl/autograd/tensor/AutogradOps.h), which dispatches to the registered `AutogradExtention` implemetations for those operations' input tensors' backends.
