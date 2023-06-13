/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/tensor.h"

int main() {
  fl::init();
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 5}, 2);
  auto c = fl::minimum(a, b);

  return 0;
}
