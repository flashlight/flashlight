/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace ::testing;
using namespace fl;

TEST(TensorReductionTest, countNonzero) {
  std::vector<int> idxs = {0, 3, 4, 7, 24, 78};
  auto a = fl::full({10, 10}, 1, fl::dtype::u32);
  for (const auto idx : idxs) {
    a(idx / 10, idx % 10) = 0;
  }

  ASSERT_TRUE(
      allClose(fl::fromScalar(a.elements() - idxs.size()), fl::countNonzero(a)));

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
      allClose(fl::fromScalar(b.elements() - 3), fl::countNonzero(b, {0, 1, 2})));
}

TEST(TensorReductionTest, amin) {
  auto a = fl::rand({4, 5, 6});
  const float val = -300;
  a(2, 3, 4) = val;
  ASSERT_EQ(fl::amin(a).shape(), Shape({}));
  ASSERT_EQ(fl::amin(a).elements(), 1);
  ASSERT_EQ(fl::amin(a).scalar<float>(), val);
  auto b = fl::rand({4, 4});
  b(1, 1) = val;
  ASSERT_EQ(fl::amin(b, {0}).shape(), Shape({4}));
  ASSERT_EQ(fl::amin(b, {0}, /* keepDims = */ true).shape(), Shape({1, 4}));
  ASSERT_EQ(fl::amin(b, {0})(1).scalar<float>(), val);
  ASSERT_EQ(fl::amin(b, {1})(1).scalar<float>(), val);
  auto q = fl::amin(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(q.shape(), Shape({}));
  ASSERT_EQ(q.elements(), 1);
  ASSERT_EQ(q.scalar<int>(), 1);

  const float v = 3.14;
  auto s = fl::amin(fl::fromScalar(v));
  ASSERT_EQ(s.shape(), Shape());
  ASSERT_EQ(s.scalar<float>(), v);
  ASSERT_EQ(fl::amin(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorReductionTest, amax) {
  auto a = fl::rand({4, 5, 6});
  const float val = 300;
  a(2, 3, 4) = val;
  ASSERT_EQ(fl::amax(a).shape(), Shape({}));
  ASSERT_EQ(fl::amax(a).elements(), 1);
  ASSERT_EQ(fl::amax(a).scalar<float>(), val);
  auto b = fl::rand({4, 4});
  b(1, 1) = val;
  ASSERT_EQ(fl::amax(b, {0}).shape(), Shape({4}));
  ASSERT_EQ(fl::amax(b, {0}, /* keepDims = */ true).shape(), Shape({1, 4}));
  ASSERT_EQ(fl::amax(b, {0})(1).scalar<float>(), val);
  ASSERT_EQ(fl::amax(b, {1})(1).scalar<float>(), val);
  auto q = fl::amax(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(q.shape(), Shape({}));
  ASSERT_EQ(q.elements(), 1);
  ASSERT_EQ(q.scalar<int>(), 1);

  const float v = 3.14;
  auto s = fl::amax(fl::fromScalar(v));
  ASSERT_EQ(s.shape(), Shape());
  ASSERT_EQ(s.scalar<float>(), v);
  ASSERT_EQ(fl::amax(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorReductionTest, argmin) {
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

TEST(TensorReductionTest, argmax) {
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

TEST(TensorReductionTest, min) {
  Tensor in = Tensor::fromVector<float>({2, 3}, {4, 8, 6, 3, 5, 9});
  Tensor values, indices;
  fl::min(values, indices, in, 0);
  ASSERT_EQ(indices.shape(), Shape({in.dim(1)}));
  ASSERT_TRUE(allClose(indices, Tensor::fromVector<unsigned>({3}, {0, 1, 0})));
  for (unsigned i = 0; i < values.elements(); ++i) {
    ASSERT_TRUE(allClose(values.flat(i), in(fl::span, i)(indices(i))));
  }

  fl::min(values, indices, in, 1);
  ASSERT_EQ(indices.shape(), Shape({in.dim(0)}));
  ASSERT_TRUE(allClose(indices, Tensor::fromVector<unsigned>({2}, {0, 1})));
  for (unsigned i = 0; i < values.elements(); ++i) {
    ASSERT_TRUE(allClose(values.flat(i), in(i)(indices(i))));
  }

  fl::min(values, indices, in, 0, /* keepDims = */ true);
  ASSERT_EQ(values.shape(), Shape({1, in.dim(1)}));

  fl::min(values, indices, in, 1, /* keepDims = */ true);
  ASSERT_EQ(values.shape(), Shape({in.dim(0), 1}));
}

TEST(TensorReductionTest, max) {
  Tensor in = Tensor::fromVector<float>({2, 3}, {4, 8, 6, 3, 5, 9});
  Tensor values, indices;
  fl::max(values, indices, in, 0);
  ASSERT_EQ(indices.shape(), Shape({in.dim(1)}));
  ASSERT_TRUE(allClose(indices, Tensor::fromVector<unsigned>({3}, {1, 0, 1})));
  for (unsigned i = 0; i < values.elements(); ++i) {
    ASSERT_TRUE(allClose(values.flat(i), in(fl::span, i)(indices(i))));
  }

  fl::max(values, indices, in, 1);
  ASSERT_EQ(indices.shape(), Shape({in.dim(0)}));
  ASSERT_TRUE(allClose(indices, Tensor::fromVector<unsigned>({2}, {1, 2})));
  for (unsigned i = 0; i < values.elements(); ++i) {
    ASSERT_TRUE(allClose(values.flat(i), in(i)(indices(i))));
  }

  fl::max(values, indices, in, 0, /* keepDims = */ true);
  ASSERT_EQ(values.shape(), Shape({1, in.dim(1)}));

  fl::max(values, indices, in, 1, /* keepDims = */ true);
  ASSERT_EQ(values.shape(), Shape({in.dim(0), 1}));
}

TEST(TensorReductionTest, cumsum) {
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

TEST(TensorReductionTest, sum) {
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
  ASSERT_EQ(q.elements(), 1);
  ASSERT_EQ(q.scalar<int>(), dim * dim * dim * dim);

  ASSERT_TRUE(allClose(
      fl::sum(fl::sum(q, {0, 1, 2}), {0}),
      fl::fromScalar(dim * dim * dim * dim, fl::dtype::s32)));
}

TEST(TensorReductionTest, mean) {
  auto r = fl::rand({8, 7, 6});
  ASSERT_NEAR(fl::mean(r).scalar<float>(), 0.5, 0.05);
  ASSERT_EQ(
      fl::mean(r, {0, 1}, /* keepDims = */ true).shape(), Shape({1, 1, 6}));

  auto s = fl::full({5, 6, 7}, 1);
  ASSERT_TRUE(allClose(fl::mean(s, {0}), fl::full({6, 7}, 1.)));

  auto a = fl::mean(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(a.shape(), Shape({}));
  ASSERT_EQ(a.elements(), 1);
  ASSERT_EQ(a.scalar<float>(), 1.);

  // TODO: fixture this
  const float v = 3.14;
  auto q = fl::mean(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.scalar<float>(), v);
  ASSERT_EQ(fl::mean(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorReductionTest, median) {
  auto a = Tensor::fromVector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_EQ(fl::median(a).scalar<float>(), 4.5);
  ASSERT_TRUE(allClose(fl::median(a, {0}), fl::fromScalar(4.5)));
  ASSERT_EQ(fl::median(fl::rand({5, 6, 7, 8}), {1, 2}).shape(), Shape({5, 8}));
  ASSERT_EQ(
      fl::median(fl::rand({5, 6, 7, 8}), {1, 2}, /* keepDims = */ true).shape(),
      Shape({5, 1, 1, 8}));

  auto b = fl::median(fl::full({5, 5, 5, 5}, 1));
  ASSERT_EQ(b.shape(), Shape({}));
  ASSERT_EQ(b.elements(), 1);
  ASSERT_EQ(b.scalar<float>(), 1.);

  const float v = 3.14;
  auto q = fl::median(fl::fromScalar(v));
  ASSERT_EQ(q.shape(), Shape());
  ASSERT_EQ(q.scalar<float>(), v);
  ASSERT_EQ(fl::median(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorReductionTest, var) {
  auto r = fl::rand({7, 8, 9});
  auto varAll = fl::var(r);
  ASSERT_NEAR(varAll.scalar<float>(), 0.08333, 0.01);
  ASSERT_EQ(varAll.shape(), Shape({}));
  ASSERT_EQ(varAll.elements(), 1);

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

TEST(TensorReductionTest, std) {
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

TEST(TensorReductionTest, norm) {
  auto r = fl::full({7, 8, 9}, 1);
  auto normAll = fl::norm(r);
  ASSERT_FLOAT_EQ(normAll.scalar<float>(), std::sqrt(7 * 8 * 9));
  ASSERT_EQ(normAll.shape(), Shape({}));
  ASSERT_EQ(normAll.elements(), 1);
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

TEST(TensorReductionTest, any) {
  using fl::dtype;
  auto t = Tensor::fromVector<unsigned>({3, 3}, {1, 0, 0, 0, 0, 0, 0, 0, 1});
  auto anyAll = fl::any(t);
  ASSERT_EQ(anyAll.shape(), Shape({}));
  ASSERT_EQ(anyAll.elements(), 1);
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
  ASSERT_EQ(q.elements(), 1);
  ASSERT_EQ(q.scalar<char>(), true);

  const float v = 3.14;
  auto r = fl::any(fl::fromScalar(v));
  ASSERT_EQ(r.shape(), Shape());
  ASSERT_TRUE(r.scalar<char>());
  ASSERT_EQ(fl::any(fl::fromScalar(v), {0}).shape(), Shape());
}

TEST(TensorReductionTest, all) {
  using fl::dtype;
  auto t = Tensor::fromVector<unsigned>({3, 3}, {1, 0, 0, 0, 0, 0, 0, 0, 1});
  auto allAll = fl::all(t);
  ASSERT_EQ(allAll.shape(), Shape({}));
  ASSERT_EQ(allAll.elements(), 1);
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
  ASSERT_EQ(q.elements(), 1);
  ASSERT_EQ(q.scalar<char>(), true);

  const float v = 3.14;
  auto a = fl::all(fl::fromScalar(v));
  ASSERT_EQ(a.shape(), Shape());
  ASSERT_TRUE(a.scalar<char>());
  ASSERT_EQ(fl::all(fl::fromScalar(v), {0}).shape(), Shape());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
