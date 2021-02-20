/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "flashlight/fl/common/DynamicBenchmark.h"
#include "flashlight/fl/common/Init.h"

namespace {

class DynamicBenchmark : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::DynamicBenchmark::setBenchmarkMode(true);
  }
};

} // namespace

TEST_F(DynamicBenchmark, OptionsStateBasic) {
  size_t maxCount = 5;
  std::vector<int> ops = {1, 2, 3, 4, 5};
  auto options =
      std::make_shared<fl::DynamicBenchmarkOptions<int>>(ops, maxCount);

  ASSERT_FALSE(options->timingsComplete());
  ASSERT_EQ(options->currentOption(), 1);
  for (size_t i = 0; i < maxCount * ops.size(); ++i) {
    options->accumulateTimeToCurrentOption(1);
  }
  ASSERT_TRUE(options->timingsComplete());
  ASSERT_EQ(options->currentOption(), 1); // best idx should never have changed
}

TEST_F(DynamicBenchmark, OptionscurrentOptionUnchangedWithNoCountIncrement) {
  std::vector<int> ops = {1, 2, 3, 4, 5};
  auto options = std::make_shared<fl::DynamicBenchmarkOptions<int>>(
      ops, /* maxCount = */ 3);

  auto state = options->currentOption();
  options->accumulateTimeToCurrentOption(3, /* incrementCount = */ false);
  options->accumulateTimeToCurrentOption(4, /* incrementCount = */ false);
  ASSERT_EQ(state, options->currentOption());
}

TEST_F(DynamicBenchmark, OptionsStateTimed) {
  size_t maxCount = 5;
  std::unordered_set<int> ops = {1, 2, 3, 4, 5};
  auto options =
      std::make_shared<fl::DynamicBenchmarkOptions<int>>(ops, maxCount);

  for (size_t i = 0; i < maxCount * ops.size(); ++i) {
    // option 4 is faster
    if (options->currentOption() == 4) {
      options->accumulateTimeToCurrentOption(1);
    } else {
      options->accumulateTimeToCurrentOption(
          10 * (i + 1), /* incrementCount = */ false);
      options->accumulateTimeToCurrentOption(10 * (i + 1));
    }
  }
  ASSERT_TRUE(options->timingsComplete());
  ASSERT_EQ(options->currentOption(), 4); // fastest
  ASSERT_EQ(options->currentOption(), 4);
}

TEST_F(DynamicBenchmark, DynamicBenchmarkSimple) {
  size_t maxCount = 5;
  std::vector<int> sleepTimes = {4, 2, 6};

  auto options =
      std::make_shared<fl::DynamicBenchmarkOptions<int>>(sleepTimes, maxCount);
  auto dynamicBench = std::make_shared<fl::DynamicBenchmark>(options);

  for (size_t i = 0; i < maxCount * sleepTimes.size(); ++i) {
    std::chrono::milliseconds sleepTime(options->currentOption());
    dynamicBench->audit(
        [sleepTime]() { std::this_thread::sleep_for(sleepTime); });
  }
  ASSERT_TRUE(options->timingsComplete());
  // sleeping for fewer miliseconds is faster
  ASSERT_EQ(options->currentOption(), 2);
}

TEST_F(DynamicBenchmark, DynamicBenchmarkDisjointLambdas) {
  size_t maxCount = 5;
  std::vector<int> sleepTimes = {4, 2, 6};

  auto options =
      std::make_shared<fl::DynamicBenchmarkOptions<int>>(sleepTimes, maxCount);
  auto dynamicBench = std::make_shared<fl::DynamicBenchmark>(options);

  for (size_t i = 0; i < maxCount * sleepTimes.size(); ++i) {
    std::chrono::milliseconds sleepTime(options->currentOption());
    dynamicBench->audit(
        [sleepTime]() { std::this_thread::sleep_for(sleepTime); },
        /* incrementCount = */ false);

    // intermediate sleep is inversely proportional to the audit sleep time:
    // 4, 2, 6 --> 18, 24, 12
    // total duration disregarding the audit is therefore:
    // 18 + 2 * 4, 24 + 2 * 2, 12 + 2 * 6 ---> 26, 28, 24
    std::chrono::milliseconds intermediateSleepTime(
        30 - (3 * options->currentOption()));
    std::this_thread::sleep_for(intermediateSleepTime);

    dynamicBench->audit(
        [sleepTime]() { std::this_thread::sleep_for(sleepTime); });
  }
  ASSERT_TRUE(options->timingsComplete());
  // option 2 is still fastest disregarding intermediate time
  ASSERT_EQ(options->currentOption(), 2);
}

TEST_F(DynamicBenchmark, DynamicBenchmarkMatmul) {
  size_t maxCount = 5;
  // n x n arrays of different sizes
  std::vector<int> arraySizes = {12, 4, 300};

  auto options =
      std::make_shared<fl::DynamicBenchmarkOptions<int>>(arraySizes, maxCount);
  auto dynamicBench = std::make_shared<fl::DynamicBenchmark>(options);

  for (size_t i = 0; i < maxCount * arraySizes.size(); ++i) {
    auto size = dynamicBench->getOptions<fl::DynamicBenchmarkOptions<int>>()
                    ->currentOption();
    dynamicBench->audit([size]() {
      auto a = af::randu({size, size});
      auto b = af::randu({size, size});
      auto c = af::matmul(a, b);
      c.eval();
    });
  }
  auto ops = dynamicBench->getOptions<fl::DynamicBenchmarkOptions<int>>();
  ASSERT_TRUE(ops->timingsComplete());
  ASSERT_EQ(ops->currentOption(), 4);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
