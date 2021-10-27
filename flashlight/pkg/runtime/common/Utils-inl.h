/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace pkg {
namespace runtime {

/**
 * Convert an arrayfire array into a std::vector.
 *
 * @param arr input array to convert
 *
 */
template <typename T>
std::vector<T> afToVector(const af::array& arr) {
  std::vector<T> vec(arr.elements());
  arr.host(vec.data());
  return vec;
}

/**
 * Convert the array in a Variable into a std::vector.
 *
 * @param var input Variables to convert
 *
 */
template <typename T>
std::vector<T> afToVector(const fl::Variable& var) {
  return afToVector<T>(var.array());
}

} // namespace runtime
} // namespace pkg
} // namespace fl
