/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/backend/af/mem/CachingMemoryManager.h"
#include "flashlight/fl/tensor/backend/af/mem/MemoryManagerInstaller.h"
using namespace fl;

TEST(MemoryInitTest, DefaultManagerInitializesCorrectType) {
  if (FL_BACKEND_CPU) {
    GTEST_SKIP() << "CachingMemoryManager is not used on CPU backend";
  }
  auto* manager = MemoryManagerInstaller::currentlyInstalledMemoryManager();
  // A non-null value means that a) a custom memory manager has been installed
  // and b) that a CachingMemoryManager has been installed which is the desired
  // default behavior.
  ASSERT_NE(dynamic_cast<CachingMemoryManager*>(manager), nullptr);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
