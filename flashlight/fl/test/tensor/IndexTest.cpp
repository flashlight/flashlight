/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
  ASSERT_EQ(t(fl::range(1, 2), fl::range(1, 2)).shape(), Shape({1}));
  ASSERT_EQ(t(fl::range(0, fl::end)).shape(), Shape({4, 4}));
  ASSERT_EQ(t(fl::range(0, fl::end, 2)).shape(), Shape({2, 4}));

  auto t2 = fl::full({5, 6, 7, 8}, 3.);
  ASSERT_EQ(t2(2, fl::range(2, 4), fl::span, 3).shape(), Shape({2, 7}));
}

TEST(IndexTest, IndexAssignment) {
  auto t = fl::full({4, 4}, 0, fl::dtype::s32);
  t(fl::span, 0) = 1;
  t(fl::span, 1) += 1;
  t(fl::span, fl::range(2, fl::end)) += 1;
  t(fl::span, fl::span) *= 7;
  t /= 7;
  ASSERT_TRUE(allClose(t, fl::full({4, 4}, 1)));

  auto a = fl::full({6, 6}, 0.);
  a(3, 4) = 4.;
  ASSERT_TRUE(allClose(a(3, 4), fl::full({1}, 4.)));
  a(2) = fl::full({6}, 8.);
  ASSERT_TRUE(allClose(a(2), fl::full({6}, 8.)));

  auto b = fl::full({3, 3}, 1.);
  auto c = b;
  b += 1;
  ASSERT_TRUE(allClose(b, fl::full({3, 3}, 2.)));
  ASSERT_TRUE(allClose(c, fl::full({3, 3}, 1.)));

  auto q = fl::full({4, 4}, 2.);
  auto r = fl::full({4}, 3.);
  q(0) = r;
  ASSERT_TRUE(allClose(q(0), r));
  ASSERT_TRUE(allClose(q(fl::range(1, fl::end)), fl::full({3, 4}, 2.)));
}

TEST(IndexTest, TensorIndex) {
  std::vector<int> idxs = {0, 1, 4, 9, 11, 13, 16, 91};
  unsigned size = idxs.size();
  auto indices = fl::full({size}, 0);
  for (int i = 0; i < size; ++i) {
    indices(i) = idxs[i];
  }
  auto a = fl::rand({100});
  auto indexed = a(indices);
  for (int i = 0; i < size; ++i) {
    ASSERT_TRUE(allClose(indexed(i), a(idxs[i])));
  }

  a(indices) = 5.;
  ASSERT_TRUE(allClose(a(indices), fl::full({size}, 5.)));
}

TEST(IndexTest, ExpressionIndex) {
  auto a = Tensor::fromVector<int>({2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_TRUE(allClose(a(a < 5), Tensor::fromVector<int>({0, 1, 2, 3, 4})));
}
