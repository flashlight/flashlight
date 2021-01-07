/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <thread>

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/dataset/datasets.h"

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

TEST(DatasetTest, DynamicRoundRobinPacker) {
  std::vector<float> length = {2, 4, 1, 2, 3, 7, 4, 3};
  auto samples = dynamicPartitionByRoundRobin(length, 0, 2, 12);
  ASSERT_EQ(samples.first.size(), 4);
  // indices which packed into 0-th thread
  ASSERT_EQ(samples.first, std::vector<int64_t>({0, 1, 2, 5}));
  ASSERT_EQ(samples.second.size(), 2);
  // sizes of batches in the 0-th thread
  ASSERT_EQ(samples.second, std::vector<int64_t>({3, 1}));

  length = {2, 4, 1, 2, 3, 7, 4, 3, 5};
  samples = dynamicPartitionByRoundRobin(length, 0, 2, 12);
  ASSERT_EQ(samples.first.size(), 4);
  // indices which packed into 0-th thread
  ASSERT_EQ(samples.first, std::vector<int64_t>({0, 1, 2, 5}));
  ASSERT_EQ(samples.second.size(), 2);
  // sizes of batches in the 0-th thread
  ASSERT_EQ(samples.second, std::vector<int64_t>({3, 1}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
