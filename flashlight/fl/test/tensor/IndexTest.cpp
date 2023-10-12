/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace ::testing;
using namespace fl;

TEST(IndexTest, range) {
  auto s1 = fl::range(3);
  ASSERT_EQ(s1.start(), 0);
  ASSERT_EQ(s1.endVal(), 3);
  ASSERT_EQ(s1.stride(), 1);

  auto s2 = fl::range(4, 5);
  ASSERT_EQ(s2.start(), 4);
  ASSERT_EQ(s2.endVal(), 5);
  ASSERT_EQ(s2.stride(), 1);

  auto s3 = fl::range(7, 8, 9);
  ASSERT_EQ(s3.stride(), 9);

  auto s4 = fl::range(1, fl::end, 2);
  ASSERT_EQ(s4.start(), 1);
  ASSERT_EQ(s4.end(), std::nullopt);
  ASSERT_EQ(s4.stride(), 2);
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
  ASSERT_EQ(fl::Index(fl::span).type(), IndexType::Span);
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
  // TODO {0, 4} once empty ranges are supported across all backends
  // ASSERT_EQ(t(fl::range(1, 1)).shape(), Shape({1, 4}));
  ASSERT_EQ(t(fl::range(1, 2)).shape(), Shape({1, 4}));
  // TODO ditto
  // ASSERT_EQ(t(fl::span, fl::range(1, 1)).shape(), Shape({4, 1}));
  ASSERT_EQ(t(fl::range(1, 2), fl::range(1, 2)).shape(), Shape({1, 1}));
  ASSERT_EQ(t(fl::range(0, fl::end)).shape(), Shape({4, 4}));
  ASSERT_EQ(t(fl::range(0, fl::end, 2)).shape(), Shape({2, 4}));

  auto t2 = fl::full({5, 6, 7, 8}, 3.);
  ASSERT_EQ(t2(2, fl::range(2, 4), fl::span, 3).shape(), Shape({2, 7}));
  ASSERT_EQ(t2(fl::span, 3, fl::span, fl::span).shape(), Shape({5, 7, 8}));
  ASSERT_EQ(
      t2(fl::span, fl::range(1, 2), fl::span, fl::span).shape(),
      Shape({5, 1, 7, 8}));
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

  auto k = fl::rand({100, 200});
  k(3) = fl::full({200}, 0.);
  ASSERT_TRUE(allClose(k(3), fl::full({200}, 0.)));

  // Weak ref
  auto g = fl::rand({3, 4, 5});
  auto gC = g.copy();
  auto gI = g(fl::span, fl::range(0, 3));
  g(fl::span, fl::range(0, 3)) += 3;
  gI -= 3;
  ASSERT_TRUE(allClose(gC(fl::span, fl::range(0, 3)), gI));

  auto x = fl::rand({5, 6, 7, 8});
  x(3) = fl::full({6, 7, 8}, 0.);
  ASSERT_TRUE(allClose(x(3), fl::full({6, 7, 8}, 0.)));
  x(fl::span, fl::span, 2) = fl::full({5, 6, 8}, 3.);
  ASSERT_TRUE(allClose(x(fl::span, fl::span, 2), fl::full({5, 6, 8}, 3.)));
  ASSERT_THROW(
      x(fl::span, fl::span, 4) -= fl::rand({5, 6, 1, 8}),
      std::invalid_argument);

  x(fl::span, fl::range(1, 3), fl::span) = fl::full({5, 2, 7, 8}, 2.);
  ASSERT_TRUE(allClose(
      x(fl::span, fl::range(1, 3), fl::span), fl::full({5, 2, 7, 8}, 2.)));

  x(fl::span, fl::arange({5}), fl::span, fl::arange({5})) =
      fl::full({5, 5, 7, 5}, 2.);
  ASSERT_TRUE(allClose(
      x(fl::span, fl::range(1, 3), fl::span), fl::full({5, 2, 7, 8}, 2.)));
}

TEST(IndexTest, IndexInPlaceOps) {
  auto a = fl::full({4, 5, 6}, 0.);
  auto b = fl::full({5, 6}, 1.);
  a(2) += b;
  ASSERT_TRUE(allClose(a(2), b));
  a(2) -= b;
  ASSERT_TRUE(allClose(a, fl::full({4, 5, 6}, 0.)));

  auto f = fl::full({1, 3, 3}, 4.);
  auto d = fl::full({3}, 6.);
  f({0, 1}) += d;
  ASSERT_TRUE(allClose(f({0, 1}), d + 4.));

  // Integral type
  auto s = fl::full({4, 5, 6}, 5, fl::dtype::s32);
  auto sA = fl::full({6}, 3, fl::dtype::s32);
  s(0, 1) += sA;
  ASSERT_TRUE(allClose(s(0, 1), sA + 5));
}

TEST(IndexTest, flat) {
  auto m = fl::rand({4, 6});
  for (unsigned i = 0; i < m.elements(); ++i) {
    ASSERT_TRUE(allClose(m.flat(i), m(i % 4, i / 4)));
  }

  auto n = fl::rand({4, 6, 8});
  for (unsigned i = 0; i < n.elements(); ++i) {
    ASSERT_TRUE(allClose(n.flat(i), n(i % 4, (i / 4) % 6, (i / (4 * 6)) % 8)));
  }

  auto a = fl::full({5, 6, 7, 8}, 9.);
  std::vector<int> testIndices = {0, 1, 4, 11, 62, 104, 288};
  for (const int i : testIndices) {
    ASSERT_EQ(a.flat(i).scalar<float>(), 9.);
  }

  a.flat(8) = 5.;
  ASSERT_EQ(a.flat(8).scalar<float>(), 5.);

  for (const int i : testIndices) {
    a.flat(i) = i + 1;
  }
  for (const int i : testIndices) {
    ASSERT_EQ(
        a(i % 5, (i / 5) % 6, (i / (5 * 6)) % 7, (i / (5 * 6 * 7)) % 8)
            .scalar<float>(),
        i + 1);
  }

  // Tensor assignment
  a.flat(32) = fl::full({1}, 7.4);
  ASSERT_TRUE(allClose(a.flatten()(32), fl::full({1}, 7.4)));
  // In-place
  a.flat(100) += 33;
  ASSERT_TRUE(allClose(a.flatten()(100), fl::full({1}, 33 + 9.)));

  // Tensor indexing
  auto indexer = Tensor::fromVector(testIndices);
  auto ref = a.flat(indexer).copy();
  ASSERT_EQ(ref.shape(), Shape({(Dim)indexer.elements()}));
  a.flat(indexer) -= 10;
  ASSERT_TRUE(allClose(a.flat(indexer), ref - 10));
  for (const int i : testIndices) {
    ASSERT_EQ(
        a(i % 5, (i / 5) % 6, (i / (5 * 6)) % 7, (i / (5 * 6 * 7)) % 8)
            .scalar<float>(),
        i + 1 - 10);
  }

  // Range flat assignment
  auto rA = fl::rand({6});
  a.flat(fl::range(1, 7)) = rA;
  ASSERT_TRUE(allClose(rA, a.flatten()(fl::range(1, 7))));

  // With leading singleton dims
  auto b = fl::rand({1, 1, 10});
  ASSERT_EQ(b.flat(fl::range(3)).shape(), Shape({3}));
  b.flat(fl::range(3)) = fl::full({3}, 6.);
  ASSERT_TRUE(allClose(b.flatten()(fl::range(3)), fl::full({3}, 6.)));
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

  // Out of range indices
  auto i = fl::arange({10}, 0, fl::dtype::u32);
  auto b = fl::rand({20, 20});
  auto ref = b;
  ASSERT_EQ(b(i).shape(), b(fl::range(10)).shape());
  ASSERT_TRUE(allClose(b(i), b(fl::range(10))));

  b(i) += 3.;
  ASSERT_TRUE(allClose(b(i), b(fl::range(10))));
  ASSERT_TRUE(allClose(b(i), (ref + 3)(i)));
  b(i) += fl::full({(Dim)i.elements(), b.dim(1)}, 10.);
  ASSERT_EQ(b(i).shape(), (ref + 13)(i).shape());
  ASSERT_TRUE(allClose(b(i), (ref + 13)(i)));

  // Tensor index a > 1D tensor
  auto c = fl::rand({10, 10, 10});
  ASSERT_EQ(c(fl::arange({5})).shape(), Shape({5, 10, 10}));
}

TEST(IndexTest, ExpressionIndex) {
  auto a = Tensor::fromVector<int>({2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_TRUE(allClose(a(a < 5), Tensor::fromVector<int>({0, 1, 2, 3, 4})));
  ASSERT_TRUE(
      allClose(a(a < 7), Tensor::fromVector<int>({0, 1, 2, 3, 4, 5, 6})));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
