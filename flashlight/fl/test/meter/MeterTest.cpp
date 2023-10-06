/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace fl;

TEST(MeterTest, EditDistanceMeter) {
  EditDistanceMeter meter;
  std::vector<int> a = {1, 2, 3, 4, 5};
  std::vector<int> b = {1, 1, 3, 3, 5, 6};
  meter.add(Tensor::fromVector(a), Tensor::fromVector(b));
  ASSERT_EQ(meter.errorRate()[0], 50.0); // 3 / 6
  ASSERT_EQ(meter.value()[0], 3); // 3 / 6
  ASSERT_LT(
      std::fabs(16.6666667 - meter.errorRate()[2]), 1e-5); // deletion = 1 / 6
  ASSERT_EQ(meter.value()[2], 1);
  ASSERT_EQ(meter.errorRate()[3], 0.0); // insertion error
  ASSERT_EQ(meter.value()[3], 0);
  ASSERT_LT(
      std::fabs(33.3333333 - meter.errorRate()[4]),
      1e-5); // substitution error = 2 / 6
  ASSERT_EQ(meter.value()[4], 2);
  // TODO{fl::Tensor}{check}
  meter.add(
      Tensor::fromBuffer({3}, a.data() + 1, MemoryLocation::Host),
      Tensor::fromVector({3}, b));
  ASSERT_LT(std::fabs(66.666666 - meter.errorRate()[0]), 1e-5); // 3 + 3 / 6 + 3
  ASSERT_EQ(meter.value()[0], 6);
}

TEST(MeterTest, FrameErrorMeter) {
  FrameErrorMeter meter;
  std::vector<int> a = {1, 2, 3, 4, 5};
  std::vector<int> b = {1, 1, 3, 3, 5, 6};
  meter.add(Tensor::fromVector(a), Tensor::fromVector({5}, b));
  ASSERT_EQ(meter.value(), 40.0); // 2 / 5
  // TODO{fl::Tensor}{check}
  meter.add(
      Tensor::fromBuffer({4}, a.data() + 1, MemoryLocation::Host),
      Tensor::fromBuffer({4}, b.data() + 2, MemoryLocation::Host));
  ASSERT_LT(std::fabs(55.5555555 - meter.value()), 1e-5); // 2 + 3 / 5 + 4
}

TEST(MeterTest, AverageValueMeter) {
  AverageValueMeter meter;
  meter.add(1.0, 0.0);
  meter.add(2.0);
  meter.add(3.0);
  meter.add(4.0);
  auto val = meter.value();
  ASSERT_EQ(val[0], 3.0);
  ASSERT_NEAR(val[1], 1.0, 1e-10);
  ASSERT_EQ(val[2], 3.0);

  std::vector<float> a = {2.0, 3.0, 4.0};
  meter.add(Tensor::fromVector(a));
  val = meter.value();
  ASSERT_EQ(val[0], 3.0);
  ASSERT_NEAR(val[1], 0.8, 1e-10);
  ASSERT_EQ(val[2], 6.0);
}

TEST(MeterTest, MSEMeter) {
  MSEMeter meter;
  std::vector<int> b = {4, 5, 6, 7, 8};
  meter.add(
      Tensor::fromVector<int>({1, 2, 3, 4, 5}),
      Tensor::fromVector<int>({4, 5, 6, 7, 8}));
  auto val = meter.value();
  ASSERT_EQ(val, 45.0);
}

TEST(MeterTest, CountMeter) {
  CountMeter meter{3};
  meter.add(0, 10);
  meter.add(1, 11);
  meter.add(0, 12);
  auto val = meter.value();
  ASSERT_EQ(val[0], 22);
  ASSERT_EQ(val[1], 11);
  ASSERT_EQ(val[2], 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
