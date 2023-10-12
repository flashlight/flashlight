/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/halide/HalideInterface.h"

#include <unordered_map>

#include "flashlight/fl/tensor/Compute.h"

#include <cublas_v2.h> // this must proceed `af/cuda.h` for some reason
#include <af/cuda.h>
#include <af/device.h>
#include <cuda.h> // Driver API needed for CUcontext

std::unordered_map<void*, fl::Tensor> memory;

/*
 * Replace Halide weakly-linked CUDA handles.
 *
 * The Halide CUDA runtime API facilitates defining hard links to handles
 * defined in libHalide by code that calls AOT-generated pipelines. These
 * include:
 * - CUDA memory alloc functions (which are linked to the FL memory manager)
 *   - halide_cuda_device_malloc -- https://git.io/JLA8X
 * - CUDA memory free functions (which are linked to the FL memory manager)
 *   - halide_cuda_device_free -- https://git.io/JLA81
 * - Getter for the CUstream (the CUDA Driver analog of cudaStream_t)
 *   - halide_cuda_get_stream -- https://git.io/JLdYv
 * - Getter for CUcontext
 *   - halide_cuda_acquire_context -- https://git.io/JLdYe
 * - Getter for the device ID.
 *   - halide_get_gpu_device -- https://git.io/JLdYf
 *
 * Defining these hard links ensures we never have memory or synchronization
 * issues between Halide pipelines that are dropped inside of any FL operations.
 *
 * In hard links associated with the CUDA driver CUcontext, CUstream, or CUDA
 * device ID  we assume that there's always a context, stream, and device ID
 * defined in the calling thread (that Flashlight has properly-initialized CUDA
 * such that one is active). Flashlight functions that change global CUDA state
 * should take care of these implicitly.
 *
 * An alternative way to accomplish this would be to define a UserContext struct
 * which holds each of these resources, then define the __user_context symbol
 * before calling into a Halide AOT-generated pipeline; LLVM should properly
 * grab this symbol off the stack: https://git.io/JLNax
 *
 * \code

   typedef struct UserContext {
     UserContext(int id, CUcontext* ctx, cudaStream_t* stream)
       : deviceId(id), cudaContext(ctx), stream(stream) {};

     int deviceId;
     CUcontext* cudaContext;
     cudaStream_t* stream;
   } UserContext;

 * \endcode
 *
 * Initializing code:
 * \code

   // Setup CUDA -- shield your eyes
   int deviceId = fl::getDevice();
   CUcontext ctx = 0;
   CUresult res = cuCtxGetCurrent(&ctx);
   if (res != CUDA_SUCCESS) throw std::runtime_error("cuCtxGetCurrent failed");
   // NOTE All ops are required to run on the default ArrayFire CUDA stream
   cudaStream_t stream = afcu::getStream(af::getDevice());
   fl::pkg::runtime::detail::UserContext userCtx(deviceId, &ctx, &stream);
   // This symbol is searched for by LLVM on the stack before
   // JMPing to a function pointer
   void* __user_context = (void*)&userCtx;

 * \endcode
 *
 * I couldn't quite get this to work, but it might work better and not have some
 * side effects that the current implementation does. Unclear.
 */
extern "C" {

int halide_cuda_device_malloc(void* /* user_context */, halide_buffer_t* buf) {
  size_t size = buf->size_in_bytes();
  auto buffer = fl::Tensor({static_cast<long long>(size)}, fl::dtype::u8);
  void* ptr = buffer.device<void>();

  buf->device = (uint64_t)ptr; // eh
  buf->device_interface = halide_cuda_device_interface();
  // This doesn't work because the public device interface API returns an
  // incomplete type. Is it needed? Who knows
  // buf->device_interface->impl->use_module();

  memory[ptr] = std::move(buffer);
  return 0;
}

int halide_cuda_device_free(void* /* user_context */, halide_buffer_t* buf) {
  void* ptr = (void*)buf->device;

  auto iter = memory.find(ptr);
  if (iter == memory.end()) {
    throw std::runtime_error(
        "halide_cuda_device_free - FL - tried to free unmanaged buffer");
  }
  iter->second.unlock();
  memory.erase(iter);

  // See above - we never call use_module(), so don't release it I suppose...
  // buf->device_interface->impl->release_module();
  buf->device_interface = nullptr;
  buf->device = 0;
  return 0;
}

int halide_cuda_acquire_context(
    void* /* user_context */,
    CUcontext* ctx,
    bool create = true) {
  CUcontext _ctx = 0;
  CUresult res = cuCtxGetCurrent(&_ctx);
  *ctx = _ctx;
  return 0;
}

int halide_cuda_get_stream(
    void* /* user_context */,
    CUcontext /* ctx */,
    CUstream* stream) {
  // NOTE All ops are required to run on the default ArrayFire CUDA stream
  *stream = (CUstream)afcu::getStream(af::getDevice());
  return 0;
}

int halide_get_gpu_device(void* /* user_context */) {
  return fl::getDevice();
}

} // extern "C"

namespace fl {
namespace pkg {
namespace halide {

std::vector<int> flToHalideDims(const Shape& dims) {
  const auto ndims = dims.ndim();
  std::vector<int> halideDims(ndims);
  for (int i = 0; i < ndims; ++i) {
    halideDims[ndims - 1 - i] = static_cast<int>(dims[i]);
  }
  return halideDims;
}

Shape halideToFlDims(const Halide::Buffer<void>& buffer) {
  const int nDims = buffer.dimensions();
  // Fastpaths
  if (nDims == 0) {
    // Scalar tensor
    return {};
  }
  for (size_t i = 0; i < nDims; ++i) {
    // Empty tensor
    if (buffer.dim(i).extent() == 0) {
      return {0};
    }
  }

  Shape out(std::vector<Dim>(nDims, 1));
  for (size_t i = 0; i < nDims; ++i) {
    auto size = static_cast<dim_t>(buffer.dim(i).extent());
    out[nDims - 1 - i] = size;
  }
  return out;
}

fl::dtype halideRuntimeTypeToFlType(halide_type_t type) {
  halide_type_code_t typeCode = type.code;
  switch (typeCode) {
    case halide_type_int:
      switch (type.bytes()) {
        case 2:
          return fl::dtype::s16;
        case 4:
          return fl::dtype::s32;
        case 8:
          return fl::dtype::s64;
      }
    case halide_type_uint:
      switch (type.bytes()) {
        case 2:
          return fl::dtype::u16;
        case 4:
          return fl::dtype::u32;
        case 8:
          return fl::dtype::u64;
      }
    case halide_type_float:
      switch (type.bytes()) {
        case 2:
          return fl::dtype::f16;
        case 4:
          return fl::dtype::f32;
        case 8:
          return fl::dtype::f64;
      }
    default:
      throw std::invalid_argument(
          "halideRuntimeTypeToFlType: unsupported or unknown Halide type");
  }
}
} // namespace halide
} // namespace pkg
} // namespace fl
