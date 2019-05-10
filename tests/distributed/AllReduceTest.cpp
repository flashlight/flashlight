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

const std::string kSkipDistributedTestsFlag = "-Distributed.*";

TEST(Distributed, AllReduce) {
  auto rank = getWorldRank();
  auto size = getWorldSize();

  Variable var(af::constant(rank, 10), false);

  allReduce(var, 2.0);

  float expected_val = size * (size - 1.0);
  ASSERT_TRUE(af::allTrue<bool>(var.array() == expected_val));
}

TEST(Distributed, InlineReducer) {
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
  auto rank = getWorldRank();
  auto size = getWorldSize();

  Variable var(af::constant(rank, 10), false);

  allReduce(var, 2.0, /* async = */ true);
  syncDistributed();

  float expected_val = size * (size - 1.0);
  ASSERT_TRUE(af::allTrue<bool>(var.array() == expected_val));
}

TEST(Distributed, AllReduceSetAsync) {
  auto rank = getWorldRank();
  auto size = getWorldSize();

  size_t vSize = (1 << 20);
  std::vector<Variable> vars;
  for (size_t i = 0; i < 5; ++i) {
    vars.push_back(Variable(af::constant(rank + 1, vSize), false));
  }

  allReduceMultiple(vars, 2.0, /* async = */ true, /* contiguous = */ true);
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
        allReduceMultiple(
            vars, 2.0, /* async = */ true, /* contiguous = */ true),
        std::runtime_error);
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
    std::cerr << "Distributed initialization failed: " << ex.what()
              << std::endl;
    testing::GTEST_FLAG(filter) = kSkipDistributedTestsFlag;
  }

  return RUN_ALL_TESTS();
}
