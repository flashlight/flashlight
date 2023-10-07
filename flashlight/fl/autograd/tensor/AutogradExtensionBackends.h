/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorExtension.h"

/*
 * Conditionally include autograd extensions
 */
#if FL_USE_CUDNN
  #include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnAutogradExtension.h"
#endif // FL_USE_CUDNN
#if FL_USE_ONEDNN
  #include "flashlight/fl/autograd/tensor/backend/onednn/OneDnnAutogradExtension.h"
#endif // FL_USE_ONEDNN

namespace fl {

/****************** Autograd Extension Registration ******************/

// TODO{fl::Tensor} -- improve macros based on compute envs
#if FL_USE_CUDNN
  #if FL_USE_ARRAYFIRE && FL_ARRAYFIRE_USE_CUDA
FL_REGISTER_TENSOR_EXTENSION(CudnnAutogradExtension, ArrayFire);
  #endif // FL_USE_ARRAYFIRE && FL_ARRAYFIRE_USE_CUDA
#endif // FL_USE_CUDNN

#if FL_USE_ONEDNN
// OneDNN backend can transparently use its autograd extension
FL_REGISTER_TENSOR_EXTENSION(OneDnnAutogradExtension, OneDnn);

  #if FL_USE_ARRAYFIRE && (FL_ARRAYFIRE_USE_CPU || FL_ARRAYFIRE_USE_OPENCL)
FL_REGISTER_TENSOR_EXTENSION(OneDnnAutogradExtension, ArrayFire);
  #endif // FL_USE_ARRAYFIRE && (FL_ARRAYFIRE_USE_CPU ||
         // FL_ARRAYFIRE_USE_OPENCL)
#endif // FL_USE_ONEDNN

} // namespace fl
