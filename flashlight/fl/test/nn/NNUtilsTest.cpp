/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

TEST(UtilsTest, Join) {
  // Empty vector
  auto empty = join({});
  ASSERT_TRUE(empty.isEmpty());

  // Single array
  auto i = fl::rand({50, 60, 70, 1});
  auto o = join({i}, -1, 3);
  ASSERT_TRUE(fl::all(o == i).asScalar<bool>());

  // no dim for batching adds singleton dims
  ASSERT_EQ(
      join({fl::rand({50, 60, 70})}, -1, 3).shape(), Shape({50, 60, 70, 1}));
  ASSERT_EQ(join({fl::rand({50, 60})}, -1, 3).shape(), Shape({50, 60, 1, 1}));

  // more than one array
  auto a = fl::full({25, 1, 300, 1}, 1);
  auto b = fl::full({20, 1, 300, 1}, 2);
  auto c = fl::full({30, 1, 300, 1}, 3);

  auto o1 = join({a, b, c});
  ASSERT_EQ(o1.shape(), Shape({30, 1, 300, 3}));
  ASSERT_TRUE(
      fl::all(
          o1(fl::range(25), fl::range(0, 1), fl::range(300), fl::range(0, 1)) ==
          a)
          .asScalar<bool>());
  ASSERT_TRUE(fl::all(
                  o1(fl::range(25, 29),
                     fl::range(0, 1),
                     fl::range(300),
                     fl::range(0, 1)) == 0)
                  .asScalar<bool>());
  ASSERT_TRUE(
      fl::all(
          o1(fl::range(20), fl::range(0, 1), fl::range(300), fl::range(1, 2)) ==
          b)
          .asScalar<bool>());
  ASSERT_TRUE(fl::all(
                  o1(fl::range(20, 29),
                     fl::range(0, 1),
                     fl::range(300),
                     fl::range(1, 2)) == 0)
                  .asScalar<bool>());
  ASSERT_TRUE(
      fl::all(
          o1(fl::range(30), fl::range(0, 1), fl::range(300), fl::range(2, 3)) ==
          c)
          .asScalar<bool>());

  auto o2 = join({a, b, c}, -1);
  ASSERT_EQ(o2.shape(), Shape({30, 1, 300, 3}));
  ASSERT_TRUE(
      fl::all(
          o2(fl::range(25), fl::range(0, 1), fl::range(300), fl::range(0, 1)) ==
          a)
          .asScalar<bool>());
  ASSERT_TRUE(fl::all(
                  o2(fl::range(25, 29),
                     fl::range(0, 1),
                     fl::range(300),
                     fl::range(0, 1)) == -1)
                  .asScalar<bool>());
  ASSERT_TRUE(
      fl::all(
          o2(fl::range(20), fl::range(0, 1), fl::range(300), fl::range(1, 2)) ==
          b)
          .asScalar<bool>());
  ASSERT_TRUE(fl::all(
                  o2(fl::range(20, 29),
                     fl::range(0, 1),
                     fl::range(300),
                     fl::range(1, 2)) == -1)
                  .asScalar<bool>());
  ASSERT_TRUE(
      fl::all(
          o2(fl::range(30), fl::range(0, 1), fl::range(300), fl::range(2, 3)) ==
          c)
          .asScalar<bool>());

  auto o3 = join({a, b, c}, -1, 1);
  ASSERT_EQ(o3.shape(), Shape({30, 3, 300, 1}));
  ASSERT_TRUE(fl::all(o3(fl::range(25), fl::range(0, 1), fl::range(300)) == a)
                  .asScalar<bool>());
  ASSERT_TRUE(
      fl::all(o3(fl::range(25, 29), fl::range(0, 1), fl::range(300)) == -1)
          .asScalar<bool>());
  ASSERT_TRUE(fl::all(o3(fl::range(20), fl::range(1, 2), fl::range(300)) == b)
                  .asScalar<bool>());
  ASSERT_TRUE(
      fl::all(o3(fl::range(20, 29), fl::range(1, 2), fl::range(300)) == -1)
          .asScalar<bool>());
  ASSERT_TRUE(fl::all(o3(fl::range(30), fl::range(2, 3), fl::range(300)) == c)
                  .asScalar<bool>());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
