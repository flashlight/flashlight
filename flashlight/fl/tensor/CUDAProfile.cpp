/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Profile.h"

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

#include "flashlight/fl/runtime/CUDAUtils.h"

namespace fl::detail {

ScopedProfiler::ScopedProfiler() {
  FL_CUDA_CHECK(cudaProfilerStart());
}

ScopedProfiler::~ScopedProfiler() {
  FL_CUDA_CHECK(cudaProfilerStop());
}

ProfileTracer::ProfileTracer(const std::string& name) {
  nvtxRangePush(name.c_str());
}

ProfileTracer::~ProfileTracer() {
  nvtxRangePop();
}

} // namespace fl
