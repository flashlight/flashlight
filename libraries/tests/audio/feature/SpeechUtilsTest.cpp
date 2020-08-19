/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "TestUtils.h"
#include "libraries/audio/feature/SpeechUtils.h"

using namespace fl::lib;

TEST(SpeechUtilsTest, SimpleMatmul) {
  /*
    A                B
    [ 2  3  4 ]       [ 2  3 ]
    [ 3  4  5 ],      [ 3  4 ]
    [ 4  5  6 ],      [ 4  5 ]
    [ 5  6  7 ],
  */
  std::vector<float> A = {2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7};
  std::vector<float> B = {2, 3, 3, 4, 4, 5};
  auto op = cblasGemm(A, B, 2, 3);
  std::vector<float> expectedOp = {29, 38, 38, 50, 47, 62, 56, 74};
  EXPECT_TRUE(compareVec(op, expectedOp, 1E-10));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
