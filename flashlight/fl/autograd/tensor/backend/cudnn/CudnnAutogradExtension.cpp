/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnAutogradExtension.h"

#include <cudnn.h>

#include "flashlight/fl/common/DynamicBenchmark.h"

namespace fl {

std::shared_ptr<fl::DynamicBenchmark>
CudnnAutogradExtension::createBenchmarkOptions() {
  return std::make_shared<fl::DynamicBenchmark>(
      std::make_shared<fl::DynamicBenchmarkOptions<KernelMode>>(
          std::vector<KernelMode>(
              {KernelMode::F32,
               KernelMode::F32_ALLOW_CONVERSION,
               KernelMode::F16}),
          fl::kDynamicBenchmarkDefaultCount));
}

bool CudnnAutogradExtension::isDataTypeSupported(const fl::dtype& dtype) const {
  switch (dtype) {
    case fl::dtype::f16:
    case fl::dtype::f32:
    case fl::dtype::f64:
      return true;
    default:
      return false;
  }
}

} // namespace fl
