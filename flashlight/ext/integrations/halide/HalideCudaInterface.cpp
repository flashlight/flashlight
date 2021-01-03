/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/CudaUtils.h"

#include <Halide.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace fl {
namespace ext {
namespace detail {

typedef struct UserContext {
  UserContext(int id, CUcontext* ctx, cudaStream_t* stream)
      : device_id(id), cuda_context(ctx), stream(stream){};

  int device_id;
  CUcontext* cuda_context;
  cudaStream_t* stream;
} UserContext;
}
}
}

/**
 * Export these weak symbols which override weak symbols defined in Halide.
 * - halide_cuda_acquire_context -- https://git.io/JLdYe
 * - halide_cuda_get_stream -- https://git.io/JLdYv
 * - halide_get_gpu_device -- https://git.io/JLdYf
 */
extern "C" {

int halide_cuda_acquire_context(
    void* userContext,
    CUcontext* ctx,
    bool create = true) {
  if (userContext != nullptr) {
    fl::ext::detail::UserContext* user_ctx =
        (fl::ext::detail::UserContext*)userContext;
    *ctx = *user_ctx->cuda_context;
  } else {
    *ctx = nullptr;
  }
  return 0;
}

int halide_cuda_get_stream(
    void* /* userContext */,
    CUcontext /* ctx */,
    CUstream* stream) {
  std::cout << "halide_cuda_get_stream" << std::endl;
  *stream = fl::cuda::getActiveStream();
  // if (userContext != nullptr) {
  //   fl::ext::detail::UserContext* user_ctx =
  //       (fl::ext::detail::UserContext*)userContext;
  //   *stream = *user_ctx->stream;
  // } else {
  //   *stream = 0;
  // }
  // return 0;
}

int halide_get_gpu_device(void* userContext) {
  if (userContext != nullptr) {
    fl::ext::detail::UserContext* user_ctx =
        (fl::ext::detail::UserContext*)userContext;
    return user_ctx->device_id;
  } else {
    return 0;
  }
}
}
