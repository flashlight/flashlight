/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace ::testing;
using namespace fl;

TEST(IndexTest, range) {
  auto s1 = fl::range(3);
  ASSERT_EQ(s1.start(), 0);
  ASSERT_EQ(s1.end(), 2);
  ASSERT_EQ(s1.stride(), 1);

  auto s2 = fl::range(4, 5);
  ASSERT_EQ(s2.start(), 4);
  ASSERT_EQ(s2.end(), 4);
  ASSERT_EQ(s2.stride(), 1);

  auto s3 = fl::range(7, 8, 9);
  ASSERT_EQ(s3.stride(), 9);
}

TEST(IndexTest, rangeEq) {
  ASSERT_EQ(fl::range(4), fl::range(4));
  ASSERT_EQ(fl::range(2, 3), fl::range(2, 3));
  ASSERT_EQ(fl::range(5, 6, 7), fl::range(5, 6, 7));
  ASSERT_NE(fl::range(5, 11, 7), fl::range(5, 6, 7));
}

TEST(IndexTest, Type) {
  using namespace detail;
  ASSERT_EQ(fl::Index(3).type(), IndexType::Literal);
  ASSERT_EQ(fl::Index(fl::range(3)).type(), IndexType::Range);
  ASSERT_EQ(fl::Index(fl::span).type(), IndexType::Range);
  ASSERT_EQ(fl::Index(fl::full({2, 2}, 4)).type(), IndexType::Tensor);
  ASSERT_TRUE(fl::Index(fl::span).isSpan());
}

TEST(IndexTest, ArrayFireMaxIndex) {
  auto t = fl::full({2, 3, 4, 5}, 6.);
  if (t.backendType() != TensorBackendType::ArrayFire) {
    GTEST_SKIP() << "Default Tensor type isn't ArrayFire";
  }
  ASSERT_THROW(t(1, 2, 3, 4, 5), std::invalid_argument);
}

TEST(IndexTest, Shape) {
  auto t = fl::full({4, 4}, 3.);
  ASSERT_EQ(t(2, 2).shape(), Shape({1}));
  ASSERT_EQ(t(2, fl::span).shape(), Shape({4}));
  ASSERT_EQ(t(2).shape(), Shape({4}));
  ASSERT_EQ(t(fl::range(3)).shape(), Shape({3, 4}));
  ASSERT_EQ(t(fl::range(1, 2)).shape(), Shape({4}));
  ASSERT_EQ(t(fl::range(0, fl::end)).shape(), Shape({4, 4}));
  ASSERT_EQ(t(fl::range(0, fl::end, 2)).shape(), Shape({2, 4}));

  auto t2 = fl::full({5, 6, 7, 8}, 3.);
  ASSERT_EQ(t2(2, fl::range(2, 4), fl::span, 3).shape(), Shape({2, 7}));
  // TODO: add more comprehensive tests
}
