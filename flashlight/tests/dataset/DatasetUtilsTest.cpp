/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <thread>

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/dataset/datasets.h"

using namespace fl;

TEST(DatasetTest, RoundRobinPacker) {
  auto samples = partitionByRoundRobin(11, 0, 2, 2);
  ASSERT_EQ(samples.size(), 6);
  ASSERT_EQ(samples, std::vector<int64_t>({0, 1, 4, 5, 8, 9}));

  samples = partitionByRoundRobin(10, 0, 2, 2);
  ASSERT_EQ(samples.size(), 5);
  ASSERT_EQ(samples, std::vector<int64_t>({0, 1, 4, 5, 8}));

  samples = partitionByRoundRobin(9, 0, 2, 2);
  ASSERT_EQ(samples.size(), 4);
  ASSERT_EQ(samples, std::vector<int64_t>({0, 1, 4, 5}));

  samples = partitionByRoundRobin(8, 0, 2, 2);
  ASSERT_EQ(samples.size(), 4);
  ASSERT_EQ(samples, std::vector<int64_t>({0, 1, 4, 5}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
