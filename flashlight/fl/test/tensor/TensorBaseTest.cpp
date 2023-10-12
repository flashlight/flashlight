/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace ::testing;
using namespace fl;

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

TEST(TensorBaseTest, CopyConstruction) {
  Shape shape{2, 2};
  auto x = fl::full(shape, 0);
  auto y = x; // actual copy (implementation may be CoW)

  ASSERT_TRUE(allClose(x, fl::full(shape, 0)));
  ASSERT_TRUE(allClose(y, fl::full(shape, 0)));
  x += 23; // affects both tensors
  ASSERT_TRUE(allClose(x, fl::full(shape, 23)));
  ASSERT_TRUE(allClose(y, fl::full(shape, 0)));
}

TEST(TensorBaseTest, MoveConstruction) {
  Shape shape{2, 2};
  auto x = fl::full(shape, 0);
  auto y = x(span, span); // view of x

  auto z = std::move(x); // `z` takes over `x`'s data
  // TODO the following line (or any read to `y`, as it seems) promotes view to
  // copy; to avoid this, we must update impl of `assign`
  // ASSERT_TRUE(allClose(y, fl::full(shape, 0)));
  ASSERT_TRUE(allClose(z, fl::full(shape, 0)));

  z += 42; // `y` is now a view of `z`, so it's affected
  ASSERT_TRUE(allClose(y, fl::full(shape, 42)));
  ASSERT_TRUE(allClose(z, fl::full(shape, 42)));
}

TEST(TensorBaseTest, AssignmentOperatorLvalueWithRvalue) {
  Shape shape{2, 2};
  auto x = fl::full({2, 2}, 0);
  auto y = x(span, span);

  // view as a lvalue cannot be used to update original tensor
  y = fl::full({2, 2}, 42); // `x` isn't affected
  y += 1; // `x` isn't affected
  ASSERT_TRUE(allClose(x, fl::full(shape, 0)));
  ASSERT_TRUE(allClose(y, fl::full(shape, 43)));
}

TEST(TensorBaseTest, AssignmentOperatorLvalueWithLvalue) {
  Shape shape{2, 2};
  auto x = fl::full({2, 2}, 0);
  auto y = x(span, span);
  auto z = fl::full({2, 2}, 1);

  y = z; // `x` is a copy of `z` now (impl may be CoW)
  y += 1; // `z` isn't affected
  ASSERT_TRUE(allClose(x, fl::full(shape, 0)));
  ASSERT_TRUE(allClose(y, fl::full(shape, 2)));
  ASSERT_TRUE(allClose(z, fl::full(shape, 1)));
}

TEST(TensorBaseTest, AssignmentOperatorRvalueWithRvalue) {
  Shape shape{2, 2};
  auto type = dtype::f32;
  auto x = fl::full({2, 2}, 0, type);
  auto y = x(span, span);

  x(0, span) = fl::full({2}, 1); // `x` is updated by copying from rhs data
  auto res = fl::Tensor::fromVector<float>(shape, {1, 0, 1, 0}, type);
  ASSERT_TRUE(allClose(x, res));
  ASSERT_TRUE(allClose(y, res));
}

TEST(TensorBaseTest, AssignmentOperatorRvalueWithLvalue) {
  Shape shape{2, 2};
  auto type = dtype::f32;
  auto x = fl::full(shape, 0, type);
  auto y = x(span, span); // view of `x`
  auto z = fl::full({2}, 1, type);

  x(span, 1) = z; // `x` is updated by copying from `z`'s data
  x += 1; // `z` isn't affected
  auto res = fl::Tensor::fromVector<float>(shape, {1, 1, 2, 2}, type);
  ASSERT_TRUE(allClose(x, res));
  ASSERT_TRUE(allClose(y, res));
  ASSERT_TRUE(allClose(z, fl::full({2}, 1, type)));
}

TEST(TensorBaseTest, Metadata) {
  int s = 9;
  auto t = fl::rand({s, s});
  ASSERT_EQ(t.elements(), s * s);
  ASSERT_FALSE(t.isEmpty());
  ASSERT_EQ(t.bytes(), s * s * sizeof(float));

  Tensor e;
  ASSERT_EQ(e.elements(), 0);
  ASSERT_TRUE(e.isEmpty());
  ASSERT_FALSE(e.isSparse());
  ASSERT_FALSE(e.isLocked());
}

TEST(TensorBaseTest, fromScalar) {
  Tensor a = fromScalar(3.14, fl::dtype::f32);
  ASSERT_EQ(a.elements(), 1);
  ASSERT_EQ(a.ndim(), 0);
  ASSERT_FALSE(a.isEmpty());
  ASSERT_EQ(a.shape(), Shape({}));
}

TEST(TensorBaseTest, string) {
  // Different backends might print tensors differently - check for consistency
  // across two identical tensors
  auto a = fl::full({3, 4, 5}, 6.);
  auto b = fl::full({3, 4, 5}, 6.);
  ASSERT_EQ(a.toString(), b.toString());

  std::stringstream ssa, ssb;
  ssa << a;
  ssb << b;
  ASSERT_EQ(ssa.str(), ssb.str());
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

  // Empty tenors
  ASSERT_EQ(fl::concatenate(0, Tensor(), Tensor()).shape(), Shape({0}));
  ASSERT_EQ(fl::concatenate(2, Tensor(), Tensor()).shape(), Shape({0, 1, 1}));
  ASSERT_EQ(
      fl::concatenate(1, fl::rand({5, 5}), Tensor()).shape(), Shape({5, 5}));

  // More tensors
  // TODO{fl::Tensor}{concat} just concat everything once we enforce
  // arbitrarily-many tensors
  const float val = 3.;
  const int axis = 0;
  auto t = fl::concatenate(
      axis,
      fl::full({4, 2}, val),
      fl::full({4, 2}, val),
      fl::full({4, 2}, val),
      fl::concatenate(
          axis,
          fl::full({4, 2}, val),
          fl::full({4, 2}, val),
          fl::full({4, 2}, val)));
  ASSERT_EQ(t.shape(), Shape({24, 2}));
  ASSERT_TRUE(allClose(t, fl::full({24, 2}, val)));
}

TEST(TensorBaseTest, nonzero) {
  std::vector<int> idxs = {0, 1, 4, 9, 11, 23, 55, 82, 91};
  auto a = fl::full({10, 10}, 1, fl::dtype::u32);
  for (const auto idx : idxs) {
    a(idx / 10, idx % 10) = 0;
  }
  auto indices = fl::nonzero(a);
  int nnz = a.elements() - idxs.size();
  ASSERT_EQ(indices.shape(), Shape({nnz}));
  ASSERT_TRUE(
      allClose(a.flatten()(indices), fl::full({nnz}, 1, fl::dtype::u32)));
}

TEST(TensorBaseTest, flatten) {
  unsigned s = 6;
  auto a = fl::full({s, s, s}, 2.);
  auto flat = a.flatten();
  ASSERT_EQ(flat.shape(), Shape({s * s * s}));
  ASSERT_TRUE(allClose(flat, fl::full({s * s * s}, 2.)));
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
      // TODO{fl::Tensor}{concat} just concat everything once we enforce
      // arbitrarily-many tensors
      fl::concatenate(
          1,
          vTiled1,
          vTiled0,
          vTiled0,
          fl::concatenate(1, vTiled1, vTiled1, vTiled0))));
}

TEST(TensorBaseTest, astype) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(a.type(), dtype::f32);
  ASSERT_EQ(a.astype(dtype::f64).type(), dtype::f64);
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
      values,
      indices,
      a,
      /* k = */ 4,
      /* axis = */ 0,
      fl::SortMode::Ascending);
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

  auto b = fl::rand({10});
  Tensor values, indices;
  fl::sort(values, indices, b, /* axis = */ 0, SortMode::Descending);
  ASSERT_TRUE(
      allClose(values, fl::sort(b, /* axis = */ 0, SortMode::Descending)));
  ASSERT_TRUE(
      allClose(fl::argsort(b, /* axis = */ 0, SortMode::Descending), indices));
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

template <typename ScalarArgType>
void assertScalarBehavior(fl::dtype type) {
  ScalarArgType scalar = 42; // small enough for any scalar type
  auto one = fl::full({1}, scalar, type);

  if (dtype_traits<ScalarArgType>::fl_type != type) {
    ASSERT_THROW(one.template scalar<ScalarArgType>(), std::invalid_argument)
        << "dtype: " << type
        << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
    return;
  }

  if ((type == fl::dtype::f16) || (type == fl::dtype::f32) ||
      (type == fl::dtype::f64)) {
    ASSERT_FLOAT_EQ(one.template scalar<ScalarArgType>(), scalar)
        << "dtype: " << type
        << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
  } else {
    ASSERT_EQ(one.template scalar<ScalarArgType>(), scalar)
        << "dtype: " << type
        << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
  }

  auto a = fl::rand({5, 6}, type);
  ASSERT_TRUE(allClose(fl::full({1}, a.scalar<ScalarArgType>(), type), a(0, 0)))
      << "dtype: " << type
      << ", ScalarArgType: " << dtype_traits<ScalarArgType>::getName();
}

TEST(TensorBaseTest, scalar) {
  auto types = {
      fl::dtype::b8,
      fl::dtype::u8,
      fl::dtype::s16,
      fl::dtype::u16,
      fl::dtype::s32,
      fl::dtype::u32,
      fl::dtype::s64,
      fl::dtype::u64,
      fl::dtype::f16,
      fl::dtype::f32,
      fl::dtype::f64};
  for (auto type : types) {
    assertScalarBehavior<char>(type);
    assertScalarBehavior<unsigned char>(type);
    assertScalarBehavior<short>(type);
    assertScalarBehavior<unsigned short>(type);
    assertScalarBehavior<int>(type);
    assertScalarBehavior<unsigned int>(type);
    assertScalarBehavior<long>(type);
    assertScalarBehavior<unsigned long>(type);
    assertScalarBehavior<long long>(type);
    assertScalarBehavior<unsigned long long>(type);
    assertScalarBehavior<float>(type);
    assertScalarBehavior<double>(type);
  }
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

TEST(TensorBaseTest, stream) {
  auto t1 = fl::rand({10, 10});
  auto t2 = -t1;
  auto t3 = t1 + t2;
  ASSERT_EQ(&t1.stream(), &t2.stream());
  ASSERT_EQ(&t1.stream(), &t3.stream());
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
  for (int i = 0; i < a.elements(); ++i) {
    ASSERT_EQ(ptr[i], a.flatten()(i).scalar<float>());
  }

  float* existingBuffer = new float[100];
  a.host(existingBuffer);
  for (int i = 0; i < a.elements(); ++i) {
    ASSERT_EQ(existingBuffer[i], a.flatten()(i).scalar<float>());
  }

  ASSERT_EQ(Tensor().host<void>(), nullptr);
}

TEST(TensorBaseTest, toHostVector) {
  auto a = fl::rand({10, 10});
  auto vec = a.toHostVector<float>();

  for (int i = 0; i < a.elements(); ++i) {
    ASSERT_EQ(vec[i], a.flatten()(i).scalar<float>());
  }

  ASSERT_EQ(Tensor().toHostVector<float>().size(), 0);
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
