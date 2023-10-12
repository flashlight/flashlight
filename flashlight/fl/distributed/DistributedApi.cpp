/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/DistributedApi.h"

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

FL_API bool isDistributedInit() {
  return detail::DistributedInfo::getInstance().isInitialized_;
}

FL_API DistributedBackend distributedBackend() {
  return detail::DistributedInfo::getInstance().backend_;
}

FL_API void
allReduce(Variable& var, double scale /* = 1.0 */, bool async /* = false */) {
  if (getWorldSize() > 1) {
    allReduce(var.tensor(), async);
  }
  var.tensor() *= scale;
}

FL_API void allReduceMultiple(
    std::vector<Variable> vars,
    double scale /* = 1.0 */,
    bool async /* = false */,
    bool contiguous /* = false */) {
  // return a vector of pointers to avoid copying
  std::vector<Tensor*> arrs;
  for (auto& var : vars) {
    arrs.push_back(&var.tensor());
  }
  if (getWorldSize() > 1) {
    allReduceMultiple(arrs, async, contiguous);
  }
  for (auto& var : vars) {
    var.tensor() *= scale;
  }
}

FL_API void barrier() {
  auto tensor = Tensor::fromVector<int>({0});
  allReduce(tensor, false);

  // This hack is to make sure `tensor` will not be optimized away by a
  // JIT during allreduce().
  fl::sum(tensor).asScalar<float>();
}

namespace detail {
/*  static */ DistributedInfo& DistributedInfo::getInstance() {
  static DistributedInfo dinfo;
  return dinfo;
}
} // namespace detail

} // namespace fl
