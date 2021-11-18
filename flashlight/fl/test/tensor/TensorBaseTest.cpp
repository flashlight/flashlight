/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

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
  ASSERT_EQ(t.shape(), Shape({0}));
  ASSERT_EQ(t.type(), fl::dtype::f32);

  Tensor u({1, 2, 3});
  ASSERT_EQ(u.shape(), Shape({1, 2, 3}));
  ASSERT_EQ(u.type(), fl::dtype::f32);
  Tensor x({0, 3});
  ASSERT_EQ(x.shape(), Shape({0, 3}));

  Tensor q(fl::dtype::f64);
  ASSERT_EQ(q.shape(), Shape({0}));
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
  ASSERT_EQ(e.size(), 0);
  ASSERT_TRUE(e.isEmpty());
  ASSERT_FALSE(e.isSparse());
  ASSERT_FALSE(e.isLocked());
}

TEST(TensorBaseTest, fromScalar) {
  Tensor a = fromScalar(3.14, fl::dtype::f32);
  ASSERT_EQ(a.size(), 1);
  ASSERT_EQ(a.ndim(), 0);
  ASSERT_FALSE(a.isEmpty());
  ASSERT_EQ(a.shape(), Shape({}));
}

TEST(TensorBaseTest, ostream) {
  // Different backends might print tensors differently - check for consistency
  // across two identical tensors
  auto a = fl::full({3, 4, 5}, 6.);
  auto b = fl::full({3, 4, 5}, 6.);
  std::stringstream ssa, ssb;
  ssa << a;
  ssb << b;
  ASSERT_EQ(ssa.str(), ssb.str());
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
  // Tensor::fromVector
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

  // Tensor::fromArray
  constexpr unsigned arrFSize = 5;
  std::array<float, arrFSize> arrF = {1, 2, 3, 4, 5};
  auto tArrF = Tensor::fromArray(arrF);
  ASSERT_EQ(tArrF.type(), fl::dtype::f32);
  ASSERT_EQ(tArrF.shape(), Shape({arrFSize}));
  auto tArrD = Tensor::fromArray({arrFSize}, arrF, fl::dtype::f64);
  ASSERT_EQ(tArrD.type(), fl::dtype::f64);

  constexpr unsigned arrISize = 8;
  std::array<unsigned, arrISize> arrI = {1, 2, 3, 4, 5, 6, 7, 8};
  auto tArrI = Tensor::fromArray(arrI);
  ASSERT_EQ(tArrI.type(), fl::dtype::u32);
  ASSERT_EQ(tArrI.shape(), Shape({arrISize}));
  auto tArrIs = Tensor::fromArray({2, 4}, arrI);
  ASSERT_EQ(tArrIs.shape(), Shape({2, 4}));
}

TEST(TensorBaseTest, reshape) {
  auto a = fl::full({4, 4}, 3.);
  auto b = fl::reshape(a, Shape({8, 2}));
  ASSERT_EQ(b.shape(), Shape({8, 2}));
  ASSERT_TRUE(allClose(a, fl::reshape(b, {4, 4})));

  ASSERT_THROW(fl::reshape(a, {}), std::exception);
}

TEST(TensorBaseTest, transpose) {
  // TODO: expand to check els
  ASSERT_TRUE(
      allClose(fl::transpose(fl::full({3, 4}, 3.)), fl::full({4, 3}, 3.)));
  ASSERT_TRUE(allClose(
      fl::transpose(fl::full({4, 5, 6, 7}, 3.), {2, 0, 1, 3}),
      fl::full({6, 4, 5, 7}, 3.)));
  ASSERT_THROW(fl::transpose(fl::rand({3, 4, 5}), {0, 1}), std::exception);
  ASSERT_THROW(
      fl::transpose(fl::rand({2, 4, 6, 8}), {1, 0, 2}), std::exception);
  ASSERT_THROW(
      fl::transpose(fl::rand({2, 4, 6, 8}), {1, 0, 2, 4}), std::exception);

  auto a = fl::rand({4});
  ASSERT_TRUE(allClose(fl::transpose(a), a));

  ASSERT_EQ(fl::transpose(fl::rand({5, 6, 7})).shape(), Shape({7, 6, 5}));
  ASSERT_EQ(fl::transpose(fl::rand({5, 6, 1, 7})).shape(), Shape({7, 1, 6, 5}));
  ASSERT_EQ(fl::transpose(fl::rand({1, 1})).shape(), Shape({1, 1}));
  ASSERT_EQ(
      fl::transpose(fl::rand({7, 2, 1, 3}), {0, 2, 1, 3}).shape(),
      Shape({7, 1, 2, 3}));
}

TEST(TensorBaseTest, tile) {
  auto a = fl::full({4, 4}, 3.);
  auto tiled = fl::tile(a, {2, 2});
  ASSERT_EQ(tiled.shape(), Shape({8, 8}));
  ASSERT_TRUE(allClose(tiled, fl::full({8, 8}, 3.)));
  ASSERT_EQ(fl::tile(a, {}).shape(), a.shape());

  auto s = fl::fromScalar(3.14);
  ASSERT_EQ(fl::tile(s, {3, 3}).shape(), Shape({3, 3}));
  ASSERT_EQ(fl::tile(s, {}).shape(), s.shape());
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
      allClose(fl::fromScalar(a.size() - idxs.size()), fl::countNonzero(a)));

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
      allClose(fl::fromScalar(b.size() - 3), fl::countNonzero(b, {0, 1, 2})));
}

TEST(TensorBaseTest, flatten) {
  unsigned s = 6;
  auto a = fl::full({s, s, s}, 2.);
  auto flat = a.flatten();
  ASSERT_EQ(flat.shape(), Shape({s * s * s}));
  ASSERT_TRUE(allClose(flat, fl::full({s * s * s}, 2.)));
}

TEST(TensorBaseTest, amin) {
  auto a = fl::rand({4, 5, 6});
  const float val = -300;
  a(2, 3, 4) = val;
  ASSERT_EQ(fl::amin(a).shape(), Shape({}));
  ASSERT_EQ(fl::amin(a).size(), 1);
  ASSERT_EQ(fl::amin(a).scalar<float>(), val);
  auto b = fl::rand({4, 4});
  b(1, 1) = val;
  ASSERT_EQ(fl::amin(b, {0}).shape(), Shape({4}));
  ASSERT_EQ(fl::amin(b, {0}, /* keepDims = */ true).shape(), Shape({1, 4}));
  ASSERT_EQ(fl::amin(b, {0})(1).scalar<float>(), val);
  ASSERT_EQ(fl::amin(b, {1})(1).scalar<float>(), val);
  auto q = fl::amin(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(q.shape(), Shape({}));
  ASSERT_EQ(q.size(), 1);
  ASSERT_EQ(q.scalar<int>(), 1);

  const float v = 3.14;
  auto s = fl::amin(fl::fromScalar(v));
  ASSERT_EQ(s.shape(), Shape());
  ASSERT_EQ(s.scalar<float>(), v);
  ASSERT_EQ(fl::amin(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorBaseTest, amax) {
  auto a = fl::rand({4, 5, 6});
  const float val = 300;
  a(2, 3, 4) = val;
  ASSERT_EQ(fl::amax(a).shape(), Shape({}));
  ASSERT_EQ(fl::amax(a).size(), 1);
  ASSERT_EQ(fl::amax(a).scalar<float>(), val);
  auto b = fl::rand({4, 4});
  b(1, 1) = val;
  ASSERT_EQ(fl::amax(b, {0}).shape(), Shape({4}));
  ASSERT_EQ(fl::amax(b, {0}, /* keepDims = */ true).shape(), Shape({1, 4}));
  ASSERT_EQ(fl::amax(b, {0})(1).scalar<float>(), val);
  ASSERT_EQ(fl::amax(b, {1})(1).scalar<float>(), val);
  auto q = fl::amax(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(q.shape(), Shape({}));
  ASSERT_EQ(q.size(), 1);
  ASSERT_EQ(q.scalar<int>(), 1);

  const float v = 3.14;
  auto s = fl::amax(fl::fromScalar(v));
  ASSERT_EQ(s.shape(), Shape());
  ASSERT_EQ(s.scalar<float>(), v);
  ASSERT_EQ(fl::amax(fl::fromScalar(v), {0}).shape(), Shape());
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

TEST(TensorBaseTest, argmin) {
  Tensor in = Tensor::fromVector<float>({2, 3}, {4, 8, 6, 3, 5, 9});
  auto a0 = fl::argmin(in, 0);
  auto a1 = fl::argmin(in, 1);

  ASSERT_EQ(a0.shape(), Shape({in.dim(1)}));
  ASSERT_EQ(a1.shape(), Shape({in.dim(0)}));
  ASSERT_TRUE(allClose(a0, Tensor::fromVector<unsigned>({3}, {0, 1, 0})));
  ASSERT_TRUE(allClose(a1, Tensor::fromVector<unsigned>({2}, {0, 1})));
  ASSERT_EQ(
      fl::argmin(in, 0, /* keepDims = */ true).shape(), Shape({1, in.dim(1)}));
  ASSERT_EQ(
      fl::argmin(in, 1, /* keepDims = */ true).shape(), Shape({in.dim(0), 1}));
}

TEST(TensorBaseTest, argmax) {
  Tensor in = Tensor::fromVector<float>({2, 3}, {4, 8, 6, 3, 5, 9});
  auto a0 = fl::argmax(in, 0);
  auto a1 = fl::argmax(in, 1);

  ASSERT_EQ(a0.shape(), Shape({in.dim(1)}));
  ASSERT_EQ(a1.shape(), Shape({in.dim(0)}));
  ASSERT_TRUE(allClose(a0, Tensor::fromVector<unsigned>({3}, {1, 0, 1})));
  ASSERT_TRUE(allClose(a1, Tensor::fromVector<unsigned>({2}, {1, 2})));
  ASSERT_EQ(
      fl::argmax(in, 0, /* keepDims = */ true).shape(), Shape({1, in.dim(1)}));
  ASSERT_EQ(
      fl::argmax(in, 1, /* keepDims = */ true).shape(), Shape({in.dim(0), 1}));
}

TEST(TensorBaseTest, min) {
  Tensor in = Tensor::fromVector<float>({2, 3}, {4, 8, 6, 3, 5, 9});
  Tensor values, indices;
  fl::min(values, indices, in, 0);
  ASSERT_EQ(indices.shape(), Shape({in.dim(1)}));
  ASSERT_TRUE(allClose(indices, Tensor::fromVector<unsigned>({3}, {0, 1, 0})));
  for (unsigned i = 0; i < values.size(); ++i) {
    ASSERT_TRUE(allClose(values.flat(i), in(fl::span, i)(indices(i))));
  }

  fl::min(values, indices, in, 1);
  ASSERT_EQ(indices.shape(), Shape({in.dim(0)}));
  ASSERT_TRUE(allClose(indices, Tensor::fromVector<unsigned>({2}, {0, 1})));
  for (unsigned i = 0; i < values.size(); ++i) {
    ASSERT_TRUE(allClose(values.flat(i), in(i)(indices(i))));
  }

  fl::min(values, indices, in, 0, /* keepDims = */ true);
  ASSERT_EQ(values.shape(), Shape({1, in.dim(1)}));

  fl::min(values, indices, in, 1, /* keepDims = */ true);
  ASSERT_EQ(values.shape(), Shape({in.dim(0), 1}));
}

TEST(TensorBaseTest, max) {
  Tensor in = Tensor::fromVector<float>({2, 3}, {4, 8, 6, 3, 5, 9});
  Tensor values, indices;
  fl::max(values, indices, in, 0);
  ASSERT_EQ(indices.shape(), Shape({in.dim(1)}));
  ASSERT_TRUE(allClose(indices, Tensor::fromVector<unsigned>({3}, {1, 0, 1})));
  for (unsigned i = 0; i < values.size(); ++i) {
    ASSERT_TRUE(allClose(values.flat(i), in(fl::span, i)(indices(i))));
  }

  fl::max(values, indices, in, 1);
  ASSERT_EQ(indices.shape(), Shape({in.dim(0)}));
  ASSERT_TRUE(allClose(indices, Tensor::fromVector<unsigned>({2}, {1, 2})));
  for (unsigned i = 0; i < values.size(); ++i) {
    ASSERT_TRUE(allClose(values.flat(i), in(i)(indices(i))));
  }

  fl::max(values, indices, in, 0, /* keepDims = */ true);
  ASSERT_EQ(values.shape(), Shape({1, in.dim(1)}));

  fl::max(values, indices, in, 1, /* keepDims = */ true);
  ASSERT_EQ(values.shape(), Shape({in.dim(0), 1}));
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

TEST(TensorBaseTest, topk) {
  auto a = fl::arange({10, 2});
  Tensor values;
  Tensor indices;
  fl::topk(values, indices, a, /* k = */ 3, /* axis = */ 0); // descending sort
  ASSERT_TRUE(
      allClose(values, Tensor::fromVector<float>({3, 2}, {9, 8, 7, 9, 8, 7})));

  fl::topk(
      values, indices, a, /* k = */ 4, /* axis = */ 0, fl::SortMode::Ascending);
  ASSERT_TRUE(allClose(
      values, Tensor::fromVector<float>({4, 2}, {0, 1, 2, 3, 0, 1, 2, 3})));
}

TEST(TensorBaseTest, sort) {
  Shape dims({10, 2});
  auto a = fl::arange(dims);
  auto sorted = fl::sort(a, /* axis = */ 0, SortMode::Descending);

  Tensor expected({dims[0]}, a.type());
  for (int i = 0; i < dims[0]; ++i) {
    expected(i) = dims[0] - i - 1;
  }
  auto tiled = fl::tile(expected, {1, 2});
  ASSERT_TRUE(allClose(sorted, tiled));

  ASSERT_TRUE(allClose(a, fl::sort(tiled, 0, SortMode::Ascending)));
}

TEST(TensorBaseTest, argsort) {
  Shape dims({10, 2});
  auto a = fl::arange(dims);
  auto sorted = fl::argsort(a, /* axis = */ 0, SortMode::Descending);

  Tensor expected({dims[0]}, fl::dtype::u32);
  for (int i = 0; i < dims[0]; ++i) {
    expected(i) = dims[0] - i - 1;
  }
  auto tiled = fl::tile(expected, {1, 2});
  ASSERT_TRUE(allClose(sorted, tiled));

  ASSERT_TRUE(allClose(tiled, fl::argsort(tiled, 0, SortMode::Ascending)));
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

TEST(TensorBaseTest, rint) {
  Shape s = {10, 10};
  auto a = fl::rand(s) - 0.5;
  ASSERT_TRUE(allClose(fl::rint(a), fl::full(s, 0.)));
  auto b = fl::rand(s) + 0.5;
  ASSERT_TRUE(allClose(fl::rint(b), fl::full(s, 1.)));
}

TEST(TensorBaseTest, cumsum) {
  int max = 30;
  auto a = fl::tile(fl::arange(1, max), {1, 2});

  auto ref = fl::arange(1, max);
  for (int i = 1; i < max - 1; ++i) {
    ref += fl::concatenate({fl::full({i}, 0), fl::arange(1, max - i)});
  }

  ASSERT_TRUE(allClose(fl::cumsum(a, 0), fl::tile(ref, {1, 2})));
  ASSERT_TRUE(allClose(
      fl::cumsum(a, 1),
      fl::concatenate(
          {fl::arange(1, max), 2 * fl::arange(1, max)}, /* axis = */ 1)));
}

TEST(TensorBaseTest, sigmoid) {
  auto a = fl::rand({10, 10});
  ASSERT_TRUE(allClose(1 / (1 + fl::exp(-a)), fl::sigmoid(a)));
}

TEST(TensorBaseTest, flip) {
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

TEST(TensorBaseTest, asContiguousTensor) {
  auto t = fl::rand({5, 6, 7, 8});
  auto indexed =
      t(fl::range(1, 4, 2),
        fl::range(0, 6, 2),
        fl::range(0, 6, 3),
        fl::range(0, 5, 3));

  auto contiguous = indexed.asContiguousTensor();
  std::vector<Dim> strides;
  unsigned stride = 1;
  for (unsigned i = 0; i < contiguous.ndim(); ++i) {
    strides.push_back(stride);
    stride *= contiguous.dim(i);
  }
  ASSERT_EQ(contiguous.strides(), Shape(strides));
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

  ASSERT_EQ(Tensor().host<void>(), nullptr);
}

TEST(TensorBaseTest, toHostVector) {
  auto a = fl::rand({10, 10});
  auto vec = a.toHostVector<float>();

  for (int i = 0; i < a.size(); ++i) {
    ASSERT_EQ(vec[i], a.flatten()(i).scalar<float>());
  }

  ASSERT_EQ(Tensor().toHostVector<float>().size(), 0);
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

TEST(TensorBaseTest, matmulShapes) {
  using T = fl::MatrixProperty;
  // Matrix/vector/scalar multiplies
  ASSERT_EQ(fl::matmul(fl::rand({10}), fl::rand({10})).shape(), Shape({1}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::Transpose).shape(),
      Shape({1}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::Transpose, T::Transpose)
          .shape(),
      Shape({1}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::None, T::Transpose).shape(),
      Shape({1}));
  ASSERT_EQ(fl::matmul(fl::rand({1, 10}), fl::rand({10})).shape(), Shape({1}));
  ASSERT_EQ(fl::matmul(fl::rand({1}), fl::rand({1, 10})).shape(), Shape({10}));
  ASSERT_EQ(
      fl::matmul(fl::rand({10}), fl::rand({10}), T::Transpose).shape(),
      Shape({1}));
  ASSERT_EQ(fl::matmul(fl::rand({3, 4}), fl::rand({4})).shape(), Shape({3}));
  ASSERT_EQ(fl::matmul(fl::rand({5}), fl::rand({5, 7})).shape(), Shape({7}));
  ASSERT_THROW(fl::matmul(fl::rand({1}), fl::rand({10})), std::exception);
  ASSERT_THROW(fl::matmul(fl::rand({3}), fl::rand({5, 7})), std::exception);

  // Batch matrix multiply
  unsigned M = 10;
  unsigned K = 12;
  unsigned N = 14;
  unsigned b2 = 2;
  unsigned b3 = 4;
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({K, N})).shape(), Shape({M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K, b2}), fl::rand({K, N, b2})).shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K, b2, b3}), fl::rand({K, N, b2, b3})).shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K, b2, b3}), fl::rand({K, N})).shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({K, N, b2, b3})).shape(),
      Shape({M, N, b2, b3}));
  // Batch matrix multiply with transpose
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M}), fl::rand({K, N}), T::Transpose).shape(),
      Shape({M, N}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({N, K}), T::None, T::Transpose)
          .shape(),
      Shape({M, N}));
  // b2 transpose
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M, b2}), fl::rand({K, N}), T::Transpose).shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K, b2}), fl::rand({N, K}), T::None, T::Transpose)
          .shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M}), fl::rand({K, N, b2}), T::Transpose).shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(fl::rand({M, K}), fl::rand({N, K, b2}), T::None, T::Transpose)
          .shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M, b2}), fl::rand({K, N, b2}), T::Transpose)
          .shape(),
      Shape({M, N, b2}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({M, K, b2}), fl::rand({N, K, b2}), T::None, T::Transpose)
          .shape(),
      Shape({M, N, b2}));
  // b2, b3 transpose
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M, b2, b3}), fl::rand({K, N}), T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({M, K, b2, b3}), fl::rand({N, K}), T::None, T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(fl::rand({K, M}), fl::rand({K, N, b2, b3}), T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({M, K}), fl::rand({N, K, b2, b3}), T::None, T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({K, M, b2, b3}), fl::rand({K, N, b2, b3}), T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));
  ASSERT_EQ(
      fl::matmul(
          fl::rand({M, K, b2, b3}),
          fl::rand({N, K, b2, b3}),
          T::None,
          T::Transpose)
          .shape(),
      Shape({M, N, b2, b3}));

  ASSERT_EQ(
      fl::matmul(
          fl::rand({256, 200, 2}),
          fl::rand({256, 200, 2}),
          T::None,
          T::Transpose)
          .shape(),
      Shape({256, 256, 2}));
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

  unsigned dim = 5;
  auto q = fl::sum(fl::full({dim, dim, dim, dim}, 1));
  ASSERT_EQ(q.shape(), Shape({}));
  ASSERT_EQ(q.size(), 1);
  ASSERT_EQ(q.scalar<int>(), dim * dim * dim * dim);

  ASSERT_TRUE(allClose(
      fl::sum(fl::sum(q, {0, 1, 2}), {0}),
      fl::fromScalar(dim * dim * dim * dim, fl::dtype::s32)));
}

TEST(TensorBaseTest, mean) {
  auto r = fl::rand({8, 7, 6});
  ASSERT_NEAR(fl::mean(r).scalar<float>(), 0.5, 0.05);
  ASSERT_EQ(
      fl::mean(r, {0, 1}, /* keepDims = */ true).shape(), Shape({1, 1, 6}));

  auto s = fl::full({5, 6, 7}, 1);
  ASSERT_TRUE(allClose(fl::mean(s, {0}), fl::full({6, 7}, 1.)));

  auto a = fl::mean(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(a.shape(), Shape({}));
  ASSERT_EQ(a.size(), 1);
  ASSERT_EQ(a.scalar<float>(), 1.);

  // TODO: fixture this
  const float v = 3.14;
  auto q = fl::mean(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.scalar<float>(), v);
  ASSERT_EQ(fl::mean(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorBaseTest, median) {
  auto a = Tensor::fromVector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_EQ(fl::median(a).scalar<float>(), 4.5);
  ASSERT_TRUE(allClose(fl::median(a, {0}), fl::fromScalar(4.5)));
  ASSERT_EQ(fl::median(fl::rand({5, 6, 7, 8}), {1, 2}).shape(), Shape({5, 8}));
  ASSERT_EQ(
      fl::median(fl::rand({5, 6, 7, 8}), {1, 2}, /* keepDims = */ true).shape(),
      Shape({5, 1, 1, 8}));

  auto b = fl::median(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(b.shape(), Shape({}));
  ASSERT_EQ(b.size(), 1);
  ASSERT_EQ(b.scalar<float>(), 1.);

  const float v = 3.14;
  auto q = fl::median(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.scalar<float>(), v);
  ASSERT_EQ(fl::median(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorBaseTest, var) {
  auto r = fl::rand({7, 8, 9});
  auto varAll = fl::var(r);
  ASSERT_NEAR(varAll.scalar<float>(), 0.08333, 0.01);
  ASSERT_EQ(varAll.shape(), Shape({}));
  ASSERT_EQ(varAll.size(), 1);

  ASSERT_EQ(
      fl::var(r, {0, 1}, /* bias = */ false, /* keepDims = */ true).shape(),
      Shape({1, 1, 9}));

  auto s = fl::full({5, 6, 7}, 1);
  ASSERT_TRUE(allClose(fl::var(s, {0}), fl::full({6, 7}, 0.)));
  auto a = fl::rand({5, 5});
  ASSERT_TRUE(allClose(fl::var(a), fl::var(a, {0, 1})));

  const float v = 3.14;
  auto q = fl::var(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.scalar<float>(), 0);
  ASSERT_EQ(fl::var(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorBaseTest, std) {
  auto r = fl::rand({7, 8, 9});
  ASSERT_NEAR(fl::std(r).scalar<float>(), 0.2886, 0.005);
  ASSERT_EQ(
      fl::std(r, {0, 1}, /* keepDims = */ true).shape(), Shape({1, 1, 9}));

  auto s = fl::full({5, 6, 7}, 1);
  ASSERT_TRUE(allClose(fl::std(s, {0}), fl::full({6, 7}, 0.)));
  ASSERT_TRUE(allClose(fl::std(s, {1}), fl::sqrt(fl::var(s, {1}))));

  const float v = 3.14;
  auto q = fl::std(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.scalar<float>(), 0);
  ASSERT_EQ(fl::std(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorBaseTest, norm) {
  auto r = fl::full({7, 8, 9}, 1);
  auto normAll = fl::norm(r);
  ASSERT_FLOAT_EQ(normAll.scalar<float>(), std::sqrt(7 * 8 * 9));
  ASSERT_EQ(normAll.shape(), Shape({}));
  ASSERT_EQ(normAll.size(), 1);
  ASSERT_FLOAT_EQ(
      fl::norm(fl::full({5, 5}, 1.)).scalar<float>(), std::sqrt(5 * 5));
  ASSERT_EQ(
      fl::norm(r, {0, 1}, /* p = */ 2, /* keepDims = */ true).shape(),
      Shape({1, 1, 9}));

  ASSERT_FLOAT_EQ(fl::norm(r, {0}).scalar<float>(), std::sqrt(7));

  const float v = 3.14;
  auto q = fl::norm(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_NEAR(q.scalar<float>(), 3.14, 1e-4);
  ASSERT_EQ(fl::norm(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorBaseTest, any) {
  using fl::dtype;
  auto t = Tensor::fromVector<unsigned>({3, 3}, {1, 0, 0, 0, 0, 0, 0, 0, 1});
  auto anyAll = fl::any(t);
  ASSERT_EQ(anyAll.shape(), Shape({}));
  ASSERT_EQ(anyAll.size(), 1);
  ASSERT_TRUE(anyAll.scalar<char>());
  ASSERT_TRUE(allClose(
      fl::any(t, {0}),
      Tensor::fromVector<unsigned>({1, 0, 1}).astype(dtype::b8)));
  ASSERT_TRUE(allClose(fl::any(t, {0, 1}), fl::fromScalar(true, dtype::b8)));
  ASSERT_FALSE(fl::any(Tensor::fromVector<unsigned>({0, 0, 0})).scalar<char>());

  auto keptDims = fl::any(
      fl::any(t, {1}, /* keepDims = */ true), {0}, /* keepDims = */ true);
  ASSERT_EQ(keptDims.shape(), Shape({1, 1}));
  ASSERT_EQ(keptDims.scalar<char>(), fl::any(t, {0, 1}).scalar<char>());
  auto q = fl::any(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(q.shape(), Shape({}));
  ASSERT_EQ(q.size(), 1);
  ASSERT_EQ(q.scalar<char>(), true);

  const float v = 3.14;
  auto r = fl::any(fl::fromScalar(v));
  ASSERT_EQ(r.shape(), Shape());
  ASSERT_TRUE(r.scalar<char>());
  ASSERT_EQ(fl::any(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorBaseTest, all) {
  using fl::dtype;
  auto t = Tensor::fromVector<unsigned>({3, 3}, {1, 0, 0, 0, 0, 0, 0, 0, 1});
  auto allAll = fl::all(t);
  ASSERT_EQ(allAll.shape(), Shape({}));
  ASSERT_EQ(allAll.size(), 1);
  ASSERT_FALSE(allAll.scalar<char>());
  ASSERT_TRUE(allClose(
      fl::all(t, {0}),
      Tensor::fromVector<unsigned>({0, 0, 0}).astype(dtype::b8)));
  ASSERT_TRUE(allClose(fl::all(t, {0, 1}), fl::fromScalar(false, dtype::b8)));
  ASSERT_TRUE(fl::all(Tensor::fromVector<unsigned>({1, 1, 1})).scalar<char>());

  auto keptDims = fl::all(
      fl::all(t, {1}, /* keepDims = */ true), {0}, /* keepDims = */ true);
  ASSERT_EQ(keptDims.shape(), Shape({1, 1}));
  ASSERT_EQ(keptDims.scalar<char>(), fl::all(t, {0, 1}).scalar<char>());
  auto q = fl::all(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(q.shape(), Shape({}));
  ASSERT_EQ(q.size(), 1);
  ASSERT_EQ(q.scalar<char>(), true);

  const float v = 3.14;
  auto a = fl::all(fl::fromScalar(v));
  ASSERT_EQ(a.shape(), Shape());
  ASSERT_TRUE(a.scalar<char>());
  ASSERT_EQ(fl::all(fl::fromScalar(v), {0}).shape(), Shape());
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
  ASSERT_EQ(fl::iota({1, 10}, {5}).shape(), Shape({5, 10}));
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
