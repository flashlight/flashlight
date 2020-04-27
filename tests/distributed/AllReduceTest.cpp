/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <exception>
#include <iostream>

#include <gtest/gtest.h>

#include "flashlight/distributed/distributed.h"

using namespace fl;

TEST(Distributed, AllReduce) {
  if (!isDistributedInit()) {
    GTEST_SKIP() << "Distributed initialization failed or not enabled.";
  }

  auto rank = getWorldRank();
  auto size = getWorldSize();

  Variable var(af::constant(rank, 10), false);

  allReduce(var, 2.0);

  float expected_val = size * (size - 1.0);
  ASSERT_TRUE(af::allTrue<bool>(var.array() == expected_val));
}

TEST(Distributed, InlineReducer) {
  if (!isDistributedInit()) {
    GTEST_SKIP() << "Distributed initialization failed or not enabled.";
  }

  auto rank = getWorldRank();
  auto size = getWorldSize();

  Variable var(af::constant(rank, 10), false);

  auto reducer = std::make_shared<InlineReducer>(1.0 / size);
  reducer->add(var);

  // The reducer scales down by a factor of 1 / size
  auto arr = var.array() * (size * 2);

  float expected_val = size * (size - 1.0);
  ASSERT_TRUE(af::allTrue<bool>(arr == expected_val));
}

TEST(Distributed, AllReduceAsync) {
  if (!isDistributedInit()) {
    GTEST_SKIP() << "Distributed initialization failed or not enabled.";
  }

  auto rank = getWorldRank();
  auto size = getWorldSize();

  Variable var(af::constant(rank, 10), false);

  allReduce(var, 2.0, /*async=*/true);
  syncDistributed();

  float expected_val = size * (size - 1.0);
  ASSERT_TRUE(af::allTrue<bool>(var.array() == expected_val));
}

TEST(Distributed, AllReduceSetAsync) {
  if (!isDistributedInit()) {
    GTEST_SKIP() << "Distributed initialization failed or not enabled.";
  }

  auto rank = getWorldRank();
  auto size = getWorldSize();

  size_t vSize = (1 << 20);
  std::vector<Variable> vars;
  for (size_t i = 0; i < 5; ++i) {
    vars.push_back(Variable(af::constant(rank + 1, vSize), false));
  }

  allReduceMultiple(vars, 2.0, /*async=*/true, /*contiguous=*/true);
  syncDistributed();

  float expected_val = size * (size + 1.0);
  for (auto var : vars) {
    ASSERT_TRUE(af::allTrue<bool>(var.array() == expected_val));
  }

  // Exceed the size of the contiguous buffer without caching, and trigger a
  // contiguous sync with a tensor that is too large
  for (size_t i = 0; i < 25; ++i) {
    vars.push_back(Variable(af::constant(rank, vSize), false));
  }
  if (size > 1) {
    ASSERT_THROW(
        allReduceMultiple(vars, 2.0, /*async=*/true, /*contiguous=*/true),
        std::runtime_error);
  }
}

TEST(Distributed, CoalescingReducer) {
  if (!isDistributedInit()) {
    GTEST_SKIP() << "Distributed initialization failed or not enabled.";
  }

  auto rank = getWorldRank();
  auto size = getWorldSize();

  auto s = std::make_shared<fl::CoalescingReducer>(
      /* scale = */ 1.0 / size, /*async=*/true, /*contiguous=*/true);

  size_t vSize = (1 << 20);
  std::vector<Variable> vars;
  for (size_t i = 0; i < 1000; ++i) {
    vars.push_back(Variable(af::constant(rank + 1, vSize), false));
  }

  for (size_t i = 0; i < vars.size(); ++i) {
    s->add(vars[i]);
    if ((i + 1) % 10 == 0) {
      s->finalize();
    }
  }

  float expected_val = size * (size + 1.0);
  for (auto var : vars) {
    // The reducer scales down by a factor of 1 / size
    auto arr = var.array() * (size * 2);
    ASSERT_TRUE(af::allTrue<bool>(arr == expected_val));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  try {
    distributedInit(
        DistributedInit::MPI,
        -1,
        -1,
        {{DistributedConstants::kMaxDevicePerNode, "8"}});
  } catch (const std::exception& ex) {
    // Don't run the test if distributed initialization fails
    std::cerr
        << "Distributed initialization failed; tests will be skipped. Reason: "
        << ex.what() << std::endl;
  }

  return RUN_ALL_TESTS();
}
