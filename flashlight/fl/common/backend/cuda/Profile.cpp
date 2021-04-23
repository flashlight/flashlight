/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/Profile.h"

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

#include "flashlight/fl/common/backend/cuda/CudaUtils.h"

namespace fl {
namespace detail {

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

} // namespace detail
} // namespace fl
