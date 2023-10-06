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

TEST(TensorUnaryOpsTest, negative) {
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 3}, 2);
  auto c = a - b;
  ASSERT_TRUE(allClose(c, -a));
  ASSERT_TRUE(allClose(c, negative(a)));
}

TEST(TensorUnaryOpsTest, logicalNot) {
  ASSERT_TRUE(allClose(
      !fl::full({3, 3}, true), fl::full({3, 3}, false).astype(dtype::b8)));
}

TEST(TensorUnaryOpsTest, clip) {
  float h = 3.;
  float l = 2.;
  Shape s = {3, 3};
  auto high = fl::full(s, h);
  auto low = fl::full(s, l);
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), low, high), high));
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), l, high), high));
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), low, h), high));
  ASSERT_TRUE(allClose(fl::clip(fl::full({3, 3}, 4.), l, h), high));
}

TEST(TensorUnaryOpsTest, roll) {
  auto t = fl::full({5, 5}, 4.);
  ASSERT_TRUE(allClose(t, fl::roll(t, /* shift = */ 3, /* axis = */ 1)));

  Shape dims({4, 5});
  auto r = fl::arange(dims);
  auto result = fl::roll(r, /* shift = */ 1, /* axis = */ 0);
  ASSERT_EQ(r.shape(), result.shape());
  ASSERT_TRUE(allClose(result(0), fl::full({dims[1]}, dims[0] - 1, r.type())));
  ASSERT_TRUE(allClose(
      result(fl::range(1, fl::end)),
      fl::arange({dims[0] - 1, dims[1]}, /* seqDim = */ 0, r.type())));
}

TEST(TensorUnaryOpsTest, isnan) {
  Shape s = {3, 3};
  ASSERT_TRUE(allClose(
      fl::isnan(fl::full(s, 1.) / 3),
      fl::full(s, false).astype(fl::dtype::b8)));
}

TEST(TensorUnaryOpsTest, isinf) {
  Shape s = {3, 3};
  ASSERT_TRUE(allClose(
      fl::isinf(fl::full(s, 1.) / 3),
      fl::full(s, false).astype(fl::dtype::b8)));
  ASSERT_TRUE(allClose(
      fl::isinf(fl::full(s, 1.) / 0.),
      fl::full(s, true).astype(fl::dtype::b8)));
}

TEST(TensorUnaryOpsTest, sign) {
  auto vals = fl::rand({5, 5}) - 0.5;
  vals(2, 2) = 0.;
  auto signs = fl::sign(vals);
  vals(vals > 0) = 1;
  vals(vals == 0) = 0;
  vals(vals < 0) = -1;
  ASSERT_TRUE(allClose(signs, vals));
}

TEST(TensorUnaryOpsTest, tril) {
  auto checkSquareTril =
      [](const Dim dim, const Tensor& res, const Tensor& in) {
        for (int i = 0; i < dim; ++i) {
          for (int j = i + 1; j < dim; ++j) {
            ASSERT_EQ(res(i, j).scalar<float>(), 0.);
          }
        }
        for (int i = 0; i < dim; ++i) {
          for (int j = 0; j < i; ++j) {
            ASSERT_TRUE(allClose(res(i, j), in(i, j)));
          }
        }
      };
  Dim dim = 10;
  auto t = fl::rand({dim, dim});
  auto out = fl::tril(t);
  checkSquareTril(dim, out, t);

  // TODO: this could be bogus behavior
  // > 2 dims
  Dim dim2 = 3;
  auto t2 = fl::rand({dim2, dim2, dim2});
  auto out2 = fl::tril(t2);
  for (unsigned i = 0; i < dim2; ++i) {
    checkSquareTril(
        dim2, out2(fl::span, fl::span, i), t2(fl::span, fl::span, i));
  }
}

TEST(TensorUnaryOpsTest, triu) {
  auto checkSquareTriu =
      [](const Dim dim, const Tensor& res, const Tensor& in) {
        for (unsigned i = 0; i < dim; ++i) {
          for (unsigned j = i + 1; j < dim; ++j) {
            ASSERT_TRUE(allClose(res(i, j), in(i, j)));
          }
        }
        for (unsigned i = 0; i < dim; ++i) {
          for (unsigned j = 0; j < i; ++j) {
            ASSERT_EQ(res(i, j).scalar<float>(), 0.);
          }
        }
      };

  int dim = 10;
  auto t = fl::rand({dim, dim});
  auto out = fl::triu(t);
  checkSquareTriu(dim, out, t);

  // TODO: this could be bogus behavior
  // > 2 dims
  int dim2 = 3;
  auto t2 = fl::rand({dim2, dim2, dim2});
  auto out2 = fl::triu(t2);
  for (int i = 0; i < dim2; ++i) {
    checkSquareTriu(
        dim2, out2(fl::span, fl::span, i), t2(fl::span, fl::span, i));
  }
}

TEST(TensorUnaryOpsTest, floor) {
  auto a = fl::rand({10, 10}) + 0.5;
  ASSERT_TRUE(allClose((a >= 1.).astype(fl::dtype::f32), fl::floor(a)));
}

TEST(TensorUnaryOpsTest, ceil) {
  auto a = fl::rand({10, 10}) + 0.5;
  ASSERT_TRUE(allClose((a >= 1).astype(fl::dtype::f32), fl::ceil(a) - 1));
}

TEST(TensorUnaryOpsTest, rint) {
  Shape s = {10, 10};
  auto a = fl::rand(s) - 0.5;
  ASSERT_TRUE(allClose(fl::rint(a), fl::full(s, 0.)));
  auto b = fl::rand(s) + 0.5;
  ASSERT_TRUE(allClose(fl::rint(b), fl::full(s, 1.)));
}

TEST(TensorUnaryOpsTest, sigmoid) {
  auto a = fl::rand({10, 10});
  ASSERT_TRUE(allClose(1 / (1 + fl::exp(-a)), fl::sigmoid(a)));
}

TEST(TensorUnaryOpsTest, flip) {
  const unsigned high = 10;
  auto a = fl::arange({high});
  auto flipped = fl::flip(a, /* dim = */ 0);
  a *= -1;
  a += (high - 1);
  ASSERT_TRUE(allClose(a, flipped));

  auto b = fl::arange({high, high}, /* seqDim = */ 0);
  ASSERT_TRUE(allClose(fl::flip(b, 1), b));
  auto c = fl::arange({high, high}, /* seqDim = */ 1);
  ASSERT_TRUE(allClose(fl::flip(c, 0), c));
}

TEST(TensorUnaryOpsTest, where) {
  // 1 0
  // 0 1
  auto cond = fl::Tensor::fromVector<char>({2, 2}, {1, 0, 0, 1});
  // 0 2
  // 1 3
  auto x = fl::Tensor::fromVector<int>({2, 2}, {0, 1, 2, 3});
  // 4 6
  // 5 7
  auto y = fl::Tensor::fromVector<int>({2, 2}, {4, 5, 6, 7});

  // 0 6
  // 5 3
  ASSERT_TRUE(allClose(
        fl::where(cond, x, y),
        fl::Tensor::fromVector<int>({2, 2}, {0, 5, 6, 3})));
  // 0 1
  // 1 3
  ASSERT_TRUE(allClose(
        fl::where(cond, x, 1.0),
        fl::Tensor::fromVector<int>({2, 2}, {0, 1, 1, 3})));
  // 2 6
  // 5 2
  ASSERT_TRUE(allClose(
        fl::where(cond, 2.0, y),
        fl::Tensor::fromVector<int>({2, 2}, {2, 5, 6, 2})));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
