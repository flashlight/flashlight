/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/integrations/halide/HalideInterface.h"
#include "flashlight/fl/common/CudaUtils.h"

#include <af/device.h>
#include <af/dim4.hpp>

#include <cuda.h> // Driver API needed for CUcontext

/*
 * Replace Halide weakly-linked CUDA handles.
 *
 * The Halide CUDA runtime API facilitates defining hard links to handles
 * defined in libHalide by code that calls AOT-generated pipelines. These
 * include:
 * - CUDA memory alloc functions (which are linked to the AF memory manager)
 *   - halide_cuda_device_malloc -- https://git.io/JLA8X
 * - CUDA memory free functions (which are linked to the AF memory manager)
 *   - halide_cuda_device_free -- https://git.io/JLA81
 * - Getter for the CUstream (the CUDA Driver analog of cudaStream_t)
 *   - halide_cuda_get_stream -- https://git.io/JLdYv
 * - Getter for CUcontext
 *   - halide_cuda_acquire_context -- https://git.io/JLdYe
 * - Getter for the device ID.
 *   - halide_get_gpu_device -- https://git.io/JLdYf
 *
 * Defining these hard links ensures we never have memory or synchronization
 * issues between Halide pipelines that are dropped inside of any AF operations.
 *
 * In hard links associated with the CUDA driver CUcontext, CUstream, or CUDA
 * device ID  we assume that there's always a context, stream, and device ID
 * defined in the calling thread (that ArrayFire has properly-initialized CUDA
 * such that one is active). ArrayFire functions that change global CUDA state
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
   cudaStream_t stream = fl::cuda::getActiveStream();
   fl::ext::detail::UserContext userCtx(deviceId, &ctx, &stream);
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
  // TODO(jacobkahn): replace me with af::allocV2 when using AF >= 3.8
  void* ptr = af::alloc(size, af::dtype::u8);
  buf->device = (uint64_t)ptr; // eh
  buf->device_interface = halide_cuda_device_interface();
  // This doesn't work because the public device interface API returns an
  // incomplete type. Is it needed? Who knows
  // buf->device_interface->impl->use_module();
  return 0;
}

int halide_cuda_device_free(void* /* user_context */, halide_buffer_t* buf) {
  // TODO(jacobkahn): replace me with af::freeV2 when using AF >= 3.8
  af::free((void*)buf->device);

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
  if (res != CUDA_SUCCESS) {
    throw std::runtime_error("Could not get from CUDA context");
  };
  *ctx = _ctx;
  return 0;
}

int halide_cuda_get_stream(
    void* /* user_context */,
    CUcontext /* ctx */,
    CUstream* stream) {
  *stream = (CUstream)fl::cuda::getActiveStream();
  return 0;
}

int halide_get_gpu_device(void* /* user_context */) {
  return fl::getDevice();
}

} // extern "C"

namespace fl {
namespace ext {

std::vector<int> afToHalideDims(const af::dim4& dims) {
  const auto ndims = dims.ndims();
  std::vector<int> halideDims(ndims);
  for (int i = 0; i < ndims; ++i) {
    halideDims[ndims - 1 - i] = static_cast<int>(dims.dims[i]);
  }
  return halideDims;
}

af::dim4 halideToAfDims(const Halide::Buffer<void>& buffer) {
  const int nDims = buffer.dimensions();
  // Fastpaths
  if (nDims == 0) {
    return af::dim4(0);
  }
  for (size_t i = 0; i < nDims; ++i) {
    if (buffer.dim(i).extent() == 0) {
      return af::dim4(0);
    }
  }

  if (nDims > 4) {
    throw std::invalid_argument(
        "getDims: Halide buffer has greater than 4 dimensions");
  }
  af::dim4 out(1, 1, 1, 1); // initialize so unfilled dims are 1, not 0
  for (size_t i = 0; i < nDims; ++i) {
    // Halide can have size zero along a dim --> convert to size 1 for AF
    auto size = static_cast<dim_t>(buffer.dim(i).extent());
    out[nDims - 1 - i] = size == 0 ? 1 : size;
  }
  return out;
}

af::dtype halideRuntimeTypeToAfType(halide_type_t type) {
  halide_type_code_t typeCode = type.code;
  switch (typeCode) {
    case halide_type_int:
      switch (type.bytes()) {
        case 2:
          return af::dtype::s16;
        case 4:
          return af::dtype::s32;
        case 8:
          return af::dtype::s64;
      }
    case halide_type_uint:
      switch (type.bytes()) {
        case 2:
          return af::dtype::u16;
        case 4:
          return af::dtype::u32;
        case 8:
          return af::dtype::u64;
      }
    case halide_type_float:
      switch (type.bytes()) {
        case 2:
          return af::dtype::f16;
        case 4:
          return af::dtype::f32;
        case 8:
          return af::dtype::f64;
      }
    default:
      throw std::invalid_argument(
          "halideRuntimeTypeToAfType: unsupported or unknown Halide type");
  }
}
} // namespace ext
} // namespace fl
