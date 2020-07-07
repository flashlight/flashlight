/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <flashlight/flashlight.h>

namespace fl {
namespace ext {

template <typename T>
std::vector<T> afToVector(const af::array& arr) {
  std::vector<T> vec(arr.elements());
  arr.host(vec.data());
  return vec;
}

template <typename T>
std::vector<T> afToVector(const fl::Variable& var) {
  return afToVector<T>(var.array());
}

template <typename T>
void syncMeter(T& mtr) {
  if (!fl::isDistributedInit()) {
    return;
  }
  af::array arr = allreduceGet(mtr);
  fl::allReduce(arr);
  allreduceSet(mtr, arr);
}
} // namespace ext
} // namespace fl