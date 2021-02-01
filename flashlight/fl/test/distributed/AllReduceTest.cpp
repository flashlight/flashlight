/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <thread>

#include <gtest/gtest.h>

#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/distributed/distributed.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

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
  // not supported for the CPU backend
  bool async = true && !FL_BACKEND_CPU;

  Variable var(af::constant(rank, 10), false);

  allReduce(var, 2.0, async);
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
  // not supported for the CPU backend
  bool async = true && !FL_BACKEND_CPU;
  bool contiguous = true && !FL_BACKEND_CPU;

  size_t vSize = (1 << 20);
  std::vector<Variable> vars;
  for (size_t i = 0; i < 5; ++i) {
    vars.push_back(Variable(af::constant(rank + 1, vSize), false));
  }

  allReduceMultiple(vars, 2.0, async, contiguous);
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

TEST(Distributed, Barrier) {
  auto rank = getWorldRank();
  auto size = getWorldSize();
  auto suffix = "_distributed_barrier_test.txt";

  // Create files
  std::this_thread::sleep_for(std::chrono::milliseconds(5000 * rank));
  auto file = lib::getTmpPath(std::to_string(rank) + suffix);
  std::ofstream stream(file);
  stream << "done";
  stream.close();

  barrier();
  for (int i = 0; i < size; i++) {
    auto checkingFile = lib::getTmpPath(std::to_string(i) + suffix);
    ASSERT_TRUE(lib::fileExists(checkingFile));
  }
  barrier();

  // Delete files
  int status = std::remove(file.data());
  if (status != 0) {
    throw std::runtime_error("Barrier test cannot delete file: " + file);
  }
  barrier();
  for (int i = 0; i < size; i++) {
    auto checkingFile = lib::getTmpPath(std::to_string(i) + suffix);
    ASSERT_TRUE(!lib::fileExists(checkingFile));
  }
}

TEST(Distributed, CoalescingReducer) {
  if (!isDistributedInit()) {
    GTEST_SKIP() << "Distributed initialization failed or not enabled.";
  }

  auto rank = getWorldRank();
  auto size = getWorldSize();

  auto s = std::make_shared<fl::CoalescingReducer>(
      /* scale = */ 1.0 / size,
      /*async=*/true && !FL_BACKEND_CPU,
      /*contiguous=*/true && !FL_BACKEND_CPU);

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
  fl::init();

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
