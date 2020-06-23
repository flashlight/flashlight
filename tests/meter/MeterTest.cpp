/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <gtest/gtest.h>

#include "flashlight/meter/meters.h"

using namespace fl;

TEST(MeterTest, EditDistanceMeter) {
  EditDistanceMeter meter;
  std::array<int, 5> a{1, 2, 3, 4, 5};
  std::array<int, 6> b{1, 1, 3, 3, 5, 6};
  meter.add(af::array(5, a.data()), af::array(6, b.data()));
  ASSERT_EQ(meter.value()[0], 50.0); // 3 / 6
  ASSERT_LT(fabs(16.6666667 - meter.value()[2]),
            1e-5); // deletion = 1 / 6
  ASSERT_EQ(meter.value()[3], 0.0); // insertion error
  ASSERT_LT(
      fabs(33.3333333 - meter.value()[4]),
      1e-5); // substitution error = 2 / 6
  meter.add(af::array(3, a.data() + 1), af::array(3, b.data()));
  ASSERT_LT(fabs(66.666666 - meter.value()[0]),
            1e-5); // 3 + 3 / 6 + 3
}

TEST(MeterTest, FrameErrorMeter) {
  FrameErrorMeter meter;
  std::array<int, 5> a{1, 2, 3, 4, 5};
  std::array<int, 6> b{1, 1, 3, 3, 5, 6};
  meter.add(af::array(5, a.data()), af::array(5, b.data()));
  ASSERT_EQ(meter.value(), 40.0); // 2 / 5
  meter.add(af::array(4, a.data() + 1), af::array(4, b.data() + 2));
  ASSERT_LT(fabs(55.5555555 - meter.value()), 1e-5); // 2 + 3 / 5 + 4
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

  std::array<float, 3> a{2.0, 3.0, 4.0};
  meter.add(af::array(3, a.data()));
  val = meter.value();
  ASSERT_EQ(val[0], 3.0);
  ASSERT_NEAR(val[1], 0.8, 1e-10);
  ASSERT_EQ(val[2], 6.0);
}

TEST(MeterTest, MSEMeter) {
  MSEMeter meter;
  std::array<int, 5> a{1, 2, 3, 4, 5};
  std::array<int, 5> b{4, 5, 6, 7, 8};
  meter.add(af::array(5, a.data()), af::array(5, b.data()));
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
  return RUN_ALL_TESTS();
}
