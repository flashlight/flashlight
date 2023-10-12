/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <stdexcept>

#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/TensorBase.h"

TEST(TensorComputeTest, sync) {
  // Testing whether a value is ready isn't meaningful since any function to
  // inspect its state will implicitly synchronize -- this test simply ensures
  // sync runs
  auto t1 = fl::full({10, 10}, 1.);
  auto t2 = fl::full({10, 10}, 2.);
  auto t3 = t1 + t2;
  fl::sync();

  int deviceId = fl::getDevice();
  auto t4 = t1 + t2 + t3;
  fl::sync(deviceId);
}

TEST(TensorComputeTest, eval) {
  // Testing whether a value is ready isn't meaningful since any function to
  // inspect its state will implicitly synchronize -- this test simply ensures
  // eval runs
  auto t1 = fl::full({10, 10}, 3.);
  auto t2 = fl::full({10, 10}, 4.);
  auto t3 = t1 * t2;
  fl::eval(t3);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
