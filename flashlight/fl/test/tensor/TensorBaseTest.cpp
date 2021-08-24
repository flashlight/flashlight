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

TEST(TensorBaseTest, DefaultBackend) {
  Tensor t;
  ASSERT_EQ(t.backendType(), TensorBackendType::ArrayFire);
}

TEST(TensorBaseTest, DefaultConstruction) {
  Tensor t;
  ASSERT_EQ(t.shape(), Shape());
  ASSERT_EQ(t.type(), fl::dtype::f32);

  Tensor u({1, 2, 3});
  ASSERT_EQ(u.shape(), Shape({1, 2, 3}));
  ASSERT_EQ(u.type(), fl::dtype::f32);

  Tensor q(fl::dtype::f64);
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.type(), fl::dtype::f64);

  Tensor v({4, 5, 6}, fl::dtype::u64);
  ASSERT_EQ(v.shape(), Shape({4, 5, 6}));
  ASSERT_EQ(v.type(), fl::dtype::u64);
}

TEST(TensorBaseTest, Metadata) {
  int s = 9;
  auto t = fl::rand({s, s});
  ASSERT_EQ(t.size(), s * s);
  ASSERT_FALSE(t.isEmpty());
  ASSERT_EQ(t.bytes(), s * s * sizeof(float));

  Tensor e;
  ASSERT_TRUE(e.isEmpty());
}

TEST(TensorBaseTest, BinaryOperators) {
  // TODO:{fl::Tensor}{testing} expand this test/add a million fixtures, etc
  auto a = fl::full({2, 2}, 1);
  auto b = fl::full({2, 2}, 2);
  auto c = fl::full({2, 2}, 3);

  ASSERT_TRUE(allClose((a == b), (b == c)));
  ASSERT_TRUE(allClose((a + b), c));
  ASSERT_TRUE(allClose((c - b), a));
  ASSERT_TRUE(allClose((c * b), fl::full({2, 2}, 6)));

  auto d = fl::full({4, 5, 6}, 6.);
  ASSERT_THROW(a + d, std::invalid_argument);
  ASSERT_THROW(a + fl::full({7, 8}, 9.), std::exception);
}

TEST(TensorBaseTest, AssignmentOperators) {
  auto a = fl::full({3, 3}, 1.);
  a += 2;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 3.)));
  a -= 1;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 2.)));
  a *= 8;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 16.)));
  a /= 4;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 4.)));

  a = fl::full({4, 4}, 7.);
  ASSERT_TRUE(allClose(a, fl::full({4, 4}, 7.)));
  auto b = a;
  ASSERT_TRUE(allClose(b, fl::full({4, 4}, 7.)));
  a = 6.;
  ASSERT_TRUE(allClose(a, fl::full({4, 4}, 6.)));

  a = fl::full({5, 6, 7}, 8.);
  ASSERT_TRUE(allClose(a, fl::full({5, 6, 7}, 8.)));
}

TEST(TensorBaseTest, CopyOperators) {
  auto a = fl::full({3, 3}, 1.);
  auto b = a;
  a += 1;
  ASSERT_TRUE(allClose(b, fl::full({3, 3}, 1.)));
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 2.)));

  auto c = a.copy();
  a += 1;
  ASSERT_TRUE(allClose(a, fl::full({3, 3}, 3.)));
  ASSERT_TRUE(allClose(c, fl::full({3, 3}, 2.)));
}

TEST(TensorBaseTest, ConstructFromData) {
  float val = 3.;
  std::vector<float> vec(100, val);
  fl::Shape s = {10, 10};
  ASSERT_TRUE(allClose(fl::Tensor::fromVector(s, vec), fl::full(s, val)));

  ASSERT_TRUE(allClose(
      fl::Tensor::fromBuffer(s, vec.data(), fl::MemoryLocation::Host),
      fl::full(s, val)));

  std::vector<float> ascending = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto t = fl::Tensor::fromVector({3, 4}, ascending);
  ASSERT_EQ(t.type(), fl::dtype::f32);
  for (int i = 0; i < ascending.size(); ++i) {
    ASSERT_FLOAT_EQ(t(i % 3, i / 3).scalar<float>(), ascending[i]);
  }

  // TODO: add fixtures/check stuff
  std::vector<int> intV = {1, 2, 3};
  ASSERT_EQ(fl::Tensor::fromVector({3}, intV).type(), fl::dtype::s32);
  ASSERT_EQ(
      fl::Tensor::fromVector<float>({5}, {0., 1., 2., 3., 4.}).type(),
      fl::dtype::f32);

  std::vector<float> flat = {0, 1, 2, 3, 4, 5, 6, 7};
  unsigned size = flat.size();
  ASSERT_EQ(fl::Tensor::fromVector(flat).shape(), Shape({size}));
}

TEST(TensorBaseTest, reshape) {
  auto a = fl::full({4, 4}, 3.);
  auto b = fl::reshape(a, Shape({8, 2}));
  ASSERT_EQ(b.shape(), Shape({8, 2}));
  ASSERT_TRUE(allClose(a, fl::reshape(b, {4, 4})));
}

TEST(TensorBaseTest, transpose) {
  // TODO: expand to check els
  ASSERT_TRUE(
      allClose(fl::transpose(fl::full({3, 4}, 3.)), fl::full({4, 3}, 3.)));
  ASSERT_TRUE(allClose(
      fl::transpose(fl::full({4, 5, 6, 7}, 3.), {2, 0, 1}),
      fl::full({6, 4, 5, 7}, 3.)));
}

TEST(TensorBaseTest, tile) {
  auto a = fl::full({4, 4}, 3.);
  auto tiled = fl::tile(a, {2, 2});
  ASSERT_EQ(tiled.shape(), Shape({8, 8}));
  ASSERT_TRUE(allClose(tiled, fl::full({8, 8}, 3.)));
}

TEST(TensorBaseTest, concatenate) {
  auto a = fl::full({3, 3}, 1.);
  auto b = fl::full({3, 3}, 2.);
  auto c = fl::full({3, 3}, 3.);
  ASSERT_TRUE(
      allClose(fl::concatenate(0, a, b, c), fl::concatenate({a, b, c})));
  auto out = fl::concatenate(0, a, b, c);
  ASSERT_EQ(out.shape(), Shape({9, 3}));
}

TEST(TensorBaseTest, nonzero) {
  std::vector<int> idxs = {0, 1, 4, 9, 11, 23, 55, 82, 91};
  auto a = fl::full({10, 10}, 1, fl::dtype::u32);
  for (const auto idx : idxs) {
    a(idx / 10, idx % 10) = 0;
  }
  auto indices = fl::nonzero(a);
  int nnz = a.size() - idxs.size();
  ASSERT_EQ(indices.shape(), Shape({nnz}));
  ASSERT_TRUE(
      allClose(a.flatten()(indices), fl::full({nnz}, 1, fl::dtype::u32)));
}

TEST(TensorBaseTest, countNonzero) {
  std::vector<int> idxs = {0, 3, 4, 7, 24, 78};
  auto a = fl::full({10, 10}, 1, fl::dtype::u32);
  for (const auto idx : idxs) {
    a(idx / 10, idx % 10) = 0;
  }

  ASSERT_TRUE(
      allClose(fl::full({1}, a.size() - idxs.size()), fl::countNonzero(a)));

  std::vector<unsigned> sizes(a.shape().dim(0));
  for (unsigned i = 0; i < a.shape().dim(0); ++i) {
    sizes[i] =
        a.shape().dim(0) - fl::sum(a(fl::span, i) == 0, {0}).scalar<unsigned>();
  }
  ASSERT_TRUE(allClose(Tensor::fromVector(sizes), Tensor::fromVector(sizes)));

  auto b = fl::full({2, 2, 2}, 1, fl::dtype::u32);
  b(0, 1, 1) = 0;
  b(1, 0, 1) = 0;
  b(1, 1, 1) = 0;
  ASSERT_TRUE(allClose(
      fl::Tensor::fromVector<unsigned>({2, 2}, {2, 2, 1, 0}),
      fl::countNonzero(b, {0})));
  ASSERT_TRUE(allClose(
      fl::Tensor::fromVector<unsigned>({2}, {4, 1}),
      fl::countNonzero(b, {0, 1})));
  ASSERT_TRUE(
      allClose(fl::full({1}, b.size() - 3), fl::countNonzero(b, {0, 1, 2})));
}

TEST(TensorBaseTest, flatten) {
  unsigned s = 6;
  auto a = fl::full({s, s, s}, 2.);
  auto flat = a.flatten();
  ASSERT_EQ(flat.shape(), Shape({s * s * s}));
  ASSERT_TRUE(allClose(flat, fl::full({s * s * s}, 2.)));
}

TEST(TensorBaseTest, minimum) {
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 3}, 2);
  auto c = fl::minimum(a, b);
  ASSERT_EQ(a.type(), c.type());
  ASSERT_TRUE(allClose(a, c));
  ASSERT_TRUE(allClose(fl::minimum(1, b).astype(a.type()), a));
  ASSERT_TRUE(allClose(fl::minimum(b, 1).astype(a.type()), a));
}

TEST(TensorBaseTest, maximum) {
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 3}, 2);
  auto c = fl::maximum(a, b);
  ASSERT_EQ(b.type(), c.type());
  ASSERT_TRUE(allClose(b, c));
  ASSERT_TRUE(allClose(fl::maximum(1, b).astype(a.type()), b));
  ASSERT_TRUE(allClose(fl::maximum(b, 1).astype(a.type()), b));
}

TEST(TensorBaseTest, negative) {
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 3}, 2);
  auto c = a - b;
  ASSERT_TRUE(allClose(c, -a));
  ASSERT_TRUE(allClose(c, negative(a)));
}

TEST(TensorBaseTest, astype) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(a.type(), dtype::f32);
  ASSERT_EQ(a.astype(dtype::f64).type(), dtype::f64);
}

TEST(TensorBaseTest, logicalNot) {
  ASSERT_TRUE(allClose(
      !fl::full({3, 3}, true), fl::full({3, 3}, false).astype(dtype::b8)));
}

TEST(TensorBaseTest, clip) {
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

TEST(TensorBaseTest, isnan) {
  Shape s = {3, 3};
  ASSERT_TRUE(allClose(
      fl::isnan(fl::full(s, 1.) / 3),
      fl::full(s, false).astype(fl::dtype::b8)));
}

TEST(TensorBaseTest, isinf) {
  Shape s = {3, 3};
  ASSERT_TRUE(allClose(
      fl::isinf(fl::full(s, 1.) / 3),
      fl::full(s, false).astype(fl::dtype::b8)));
  ASSERT_TRUE(allClose(
      fl::isinf(fl::full(s, 1.) / 0.),
      fl::full(s, true).astype(fl::dtype::b8)));
}

TEST(TensorBaseTest, sign) {
  auto vals = fl::rand({5, 5}) - 0.5;
  vals(2, 2) = 0.;
  auto signs = fl::sign(vals);
  vals(vals > 0) = 1;
  vals(vals == 0) = 0;
  vals(vals < 0) = -1;
  ASSERT_TRUE(allClose(signs, vals));
}

TEST(TensorBaseTest, tril) {
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
  int dim = 10;
  auto t = fl::rand({dim, dim});
  auto out = fl::tril(t);
  checkSquareTril(dim, out, t);

  // TODO: this could be bogus behavior
  // > 2 dims
  int dim2 = 3;
  auto t2 = fl::rand({dim2, dim2, dim2});
  auto out2 = fl::tril(t2);
  for (int i = 0; i < dim2; ++i) {
    checkSquareTril(
        dim2, out2(fl::span, fl::span, i), t2(fl::span, fl::span, i));
  }
}

TEST(TensorBaseTest, triu) {
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

TEST(TensorBaseTest, where) {
  auto a = Tensor::fromVector<int>({2, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto out = fl::where(a < 5, a, a * 10);
  a(a >= 5) *= 10;
  ASSERT_TRUE(allClose(out, a));
  auto outC = fl::where(a < 5, a, 3);
  a(a >= 5) = 3;
  ASSERT_TRUE(allClose(outC, a));
  auto outC2 = fl::where(a < 5, 3, a);
  a(a < 5) = 3;
  ASSERT_TRUE(allClose(outC2, a));

  // non b8-type vector throws
  EXPECT_THROW(
      fl::where((a < 5).astype(fl::dtype::f32), a, a * 10), std::exception);
}

TEST(TensorBaseTest, power) {
  auto a = fl::full({3, 3}, 2.);
  auto b = fl::full({3, 3}, 2.);
  ASSERT_TRUE(allClose(fl::power(a, b), a * b));
}

TEST(TensorBaseTest, powerDouble) {
  auto a = fl::full({3, 3}, 2.);
  ASSERT_TRUE(allClose(fl::power(a, 3), a * a * a));
}

TEST(TensorBaseTest, floor) {
  auto a = fl::rand({10, 10}) + 0.5;
  ASSERT_TRUE(allClose((a >= 1.).astype(fl::dtype::f32), fl::floor(a)));
}

TEST(TensorBaseTest, ceil) {
  auto a = fl::rand({10, 10}) + 0.5;
  ASSERT_TRUE(allClose((a >= 1).astype(fl::dtype::f32), fl::ceil(a) - 1));
}

TEST(TensorBaseTest, sigmoid) {
  auto a = fl::rand({10, 10});
  ASSERT_TRUE(allClose(1 / (1 + fl::exp(-a)), fl::sigmoid(a)));
}

TEST(TensorBaseTest, scalar) {
  // TODO: exhaustively test all types + fixture
  float val = 8.47;
  auto one = fl::full({1}, val);
  ASSERT_FLOAT_EQ(one.scalar<float>(), val);

  auto a = fl::rand({5, 6});
  ASSERT_TRUE(allClose(fl::full({1}, a.scalar<float>()), a(0, 0)));

  ASSERT_THROW(a.scalar<int>(), std::invalid_argument);
}

TEST(TensorBaseTest, isContiguous) {
  // Contiguous by default
  auto a = fl::rand({10, 10});
  ASSERT_TRUE(a.isContiguous());
}

TEST(TensorBaseTest, strides) {
  auto t = fl::rand({10, 10});
  ASSERT_EQ(t.strides(), Shape({1, 10}));
}

TEST(TensorBaseTest, host) {
  auto a = fl::rand({10, 10});

  float* ptr = a.host<float>();
  for (int i = 0; i < a.size(); ++i) {
    ASSERT_EQ(ptr[i], a.flatten()(i).scalar<float>());
  }

  float* existingBuffer = new float[100];
  a.host(existingBuffer);
  for (int i = 0; i < a.size(); ++i) {
    ASSERT_EQ(existingBuffer[i], a.flatten()(i).scalar<float>());
  }
}

TEST(TensorBaseTest, toHostVector) {
  auto a = fl::rand({10, 10});
  auto vec = a.toHostVector<float>();

  for (int i = 0; i < a.size(); ++i) {
    ASSERT_EQ(vec[i], a.flatten()(i).scalar<float>());
  }
}

TEST(TensorBaseTest, matmul) {
  // TODO: test tensors with order > 2

  // Reference impl
  auto matmulRef = [](const Tensor& lhs, const Tensor& rhs) {
    // (M x N) x (N x K) --> (M x K)
    int M = lhs.dim(0);
    int N = lhs.dim(1);
    int K = rhs.dim(1);

    auto out = fl::full({M, K}, 0.);

    for (unsigned i = 0; i < M; ++i) {
      for (unsigned j = 0; j < K; ++j) {
        for (unsigned k = 0; k < N; ++k) {
          out(i, j) += lhs(i, k) * rhs(k, j);
        }
      }
    }
    return out;
  };

  int i = 10;
  int j = 20;
  int k = 12;

  auto a = fl::rand({i, j});
  auto b = fl::rand({j, k});
  auto ref = matmulRef(a, b);
  ASSERT_TRUE(allClose(fl::matmul(a, b), ref));
  ASSERT_TRUE(allClose(
      fl::matmul(
          a,
          fl::transpose(b),
          fl::MatrixProperty::None,
          fl::MatrixProperty::Transpose),
      ref));
  ASSERT_TRUE(allClose(
      fl::matmul(fl::transpose(a), b, fl::MatrixProperty::Transpose), ref));
}

TEST(TensorBaseTest, sum) {
  auto t = fl::full({3, 4, 5, 6}, 1.0);
  ASSERT_TRUE(allClose(fl::sum(t, {0}), fl::full({4, 5, 6}, 3.0)));
  ASSERT_TRUE(
      allClose(fl::sum(t, {1, 2}), fl::full({3, 6}, 4 * 5, fl::dtype::f32)));
  auto res = fl::sum(
      fl::sum(t, {2}, /* keepDims = */ true), {1}, /* keepDims = */ true);
  ASSERT_EQ(res.shape(), Shape({t.dim(0), 1, 1, t.dim(3)}));
  ASSERT_TRUE(
      allClose(fl::reshape(res, {t.dim(0), t.dim(3)}), fl::sum(t, {2, 1})));
}

TEST(TensorBaseTest, median) {
  auto a = Tensor::fromVector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_EQ(fl::median<double>(a), 4.5);
  ASSERT_TRUE(allClose(fl::median(a, {0}), fl::full({1}, 4.5)));
  ASSERT_EQ(fl::median(fl::rand({5, 6, 7, 8}), {1, 2}).shape(), Shape({5, 8}));
  ASSERT_EQ(
      fl::median(fl::rand({5, 6, 7, 8}), {1, 2}, /* keepDims = */ true).shape(),
      Shape({5, 1, 1, 8}));
}

TEST(TensorBaseTest, any) {
  using fl::dtype;
  auto t = Tensor::fromVector<unsigned>({3, 3}, {1, 0, 0, 0, 0, 0, 0, 0, 1});
  ASSERT_TRUE(allClose(
      fl::any(t, {0}),
      Tensor::fromVector<unsigned>({1, 0, 1}).astype(dtype::b8)));
  ASSERT_TRUE(allClose(
      fl::any(t, {0, 1}), Tensor::fromVector<unsigned>({1}).astype(dtype::b8)));
  ASSERT_TRUE(fl::any(t));
  ASSERT_FALSE(fl::any(Tensor::fromVector<unsigned>({0, 0, 0})));

  auto keptDims = fl::any(
      fl::any(t, {1}, /* keepDims = */ true), {0}, /* keepDims = */ true);
  ASSERT_EQ(keptDims.shape(), Shape({1, 1}));
  ASSERT_EQ(
      keptDims.astype(dtype::s32).scalar<int>(),
      fl::any(t, {0, 1}).astype(dtype::s32).scalar<int>());
}

TEST(TensorBaseTest, all) {
  using fl::dtype;
  auto t = Tensor::fromVector<unsigned>({3, 3}, {1, 0, 0, 0, 0, 0, 0, 0, 1});
  ASSERT_TRUE(allClose(
      fl::all(t, {0}),
      Tensor::fromVector<unsigned>({0, 0, 0}).astype(dtype::b8)));
  ASSERT_TRUE(allClose(
      fl::all(t, {0, 1}), Tensor::fromVector<unsigned>({0}).astype(dtype::b8)));
  ASSERT_FALSE(fl::all(t));
  ASSERT_TRUE(fl::all(Tensor::fromVector<unsigned>({1, 1, 1})));
  auto keptDims = fl::all(
      fl::all(t, {1}, /* keepDims = */ true), {0}, /* keepDims = */ true);
  ASSERT_EQ(keptDims.shape(), Shape({1, 1}));
  ASSERT_EQ(
      keptDims.astype(dtype::s32).scalar<int>(),
      fl::all(t, {0, 1}).astype(dtype::s32).scalar<int>());
}

TEST(TensorBaseTest, arange) {
  // Range/step overload
  ASSERT_TRUE(
      allClose(fl::arange(2, 10, 2), Tensor::fromVector<int>({2, 4, 6, 8})));
  ASSERT_TRUE(
      allClose(fl::arange(0, 6), Tensor::fromVector<int>({0, 1, 2, 3, 4, 5})));
  ASSERT_TRUE(allClose(
      fl::arange(0., 1.22, 0.25),
      Tensor::fromVector<float>({0., 0.25, 0.5, 0.75})));
  ASSERT_TRUE(allClose(
      fl::arange(0., 4.1), Tensor::fromVector<float>({0., 1., 2., 3.})));

  // Shape overload
  auto v = Tensor::fromVector<float>({0., 1., 2., 3.});
  ASSERT_TRUE(allClose(fl::arange({4}), v));

  ASSERT_TRUE(allClose(fl::arange({4, 5}), fl::tile(v, {1, 5})));
  ASSERT_EQ(fl::arange({4, 5}, 1).shape(), Shape({4, 5}));
  ASSERT_TRUE(allClose(
      fl::arange({4, 5}, 1),
      fl::tile(
          fl::reshape(Tensor::fromVector<float>({0., 1., 2., 3., 4.}), {1, 5}),
          {4})));
  ASSERT_EQ(fl::arange({2, 6}, 0, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(TensorBaseTest, iota) {
  ASSERT_TRUE(allClose(
      fl::iota({5, 3}, {1, 2}),
      fl::tile(fl::reshape(fl::arange({15}), {5, 3}), {1, 2})));
  ASSERT_EQ(fl::iota({2, 2}, {2, 2}, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(TensorBaseTest, pad) {
  auto t = fl::rand({5, 2});
  auto zeroPadded = fl::pad(t, {{1, 2}, {3, 4}});
  auto zeroTest = fl::concatenate(
      1,
      fl::full({8, 3}, 0.),
      fl::concatenate(0, fl::full({1, 2}, 0.), t, fl::full({2, 2}, 0.)),
      fl::full({8, 4}, 0.));
  ASSERT_TRUE(allClose(zeroPadded, zeroTest));

  auto edgePadded = fl::pad(t, {{1, 1}, {2, 2}}, PadType::Edge);
  auto vertTiled = fl::concatenate(
      0,
      fl::reshape(t(0, fl::span), {1, 2}),
      t,
      fl::reshape(t(t.dim(0) - 1, fl::span), {1, 2}));
  auto vTiled0 = vertTiled(fl::span, 0);
  auto vTiled1 = vertTiled(fl::span, 1);
  ASSERT_TRUE(allClose(
      edgePadded,
      fl::concatenate(
          1, fl::tile(vTiled0, {1, 3}), fl::tile(vTiled1, {1, 3}))));

  auto symmetricPadded = fl::pad(t, {{1, 1}, {2, 2}}, PadType::Symmetric);
  ASSERT_TRUE(allClose(
      symmetricPadded,
      fl::concatenate(
          1, vTiled1, vTiled0, vTiled0, vTiled1, vTiled1, vTiled0)));
}
