/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <future>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/lib/common/System.h"

using namespace fl::lib;

TEST(SystemTest, PathsConcat) {
  auto path1 = pathsConcat("/tmp/", "test.wav");
  auto path2 = pathsConcat("/tmp", "test.wav");
  ASSERT_EQ(path1, "/tmp/test.wav");
  ASSERT_EQ(path2, "/tmp/test.wav");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

// Resolve directory for sample dictionary
#ifdef FL_DICTIONARY_TEST_DIR
  loadPath = FL_DICTIONARY_TEST_DIR;
#endif

  return RUN_ALL_TESTS();
}
