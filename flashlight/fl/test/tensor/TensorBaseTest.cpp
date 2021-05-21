/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

// TODO: remove me when not ArrayFire specific
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

using namespace ::testing;
using namespace fl;

/*
 * Below this point are ArrayFire-specific implementation tests. They should be
 * [re]moved to a specific ArrayFire backend implementation (along with other
 * flashlight/fl/tensor assets once migration is complete).
 */

namespace {
// TODO:fl::Tensor {testing} copied from fl/common/Utils for testing since
// fl/tensor can't depend on fl/common with Buck. Move this to some other
// location later.
bool allClose(
    const af::array& a,
    const af::array& b,
    double absTolerance = 1e-5) {
  if (a.type() != b.type()) {
    return false;
  }
  if (a.dims() != b.dims()) {
    return false;
  }
  if (a.isempty() && b.isempty()) {
    return true;
  }
  return af::max<double>(af::abs(a - b)) < absTolerance;
}

bool allClose(
    const fl::Tensor& a,
    const fl::Tensor& b,
    double absTolerance = 1e-5) {
  return allClose(toArray(a), toArray(b), absTolerance);
}
} // namespace

TEST(TensorBaseTest, DefaultBackend) {
  Tensor t;
  ASSERT_EQ(t.backend(), TensorBackend::ArrayFire);
}

TEST(TensorBaseTest, AfRefCountBasic) {
  // Sanity check that af::arrays moved into fl::Tensors don't have their
  // refcount inrcremented/show proper usage of refs in tensor ops
  int refCount = 0;
  auto a = af::constant(1, {2, 2});
  af_get_data_ref_count(&refCount, a.get());
  ASSERT_EQ(refCount, 1);

  auto tensor = toTensor<ArrayFireTensor>(std::move(a));
  auto& aRef = toArray(tensor);
  af_get_data_ref_count(&refCount, aRef.get());
  ASSERT_EQ(refCount, 1);
  // Sanity check copying bumps things
  auto aNoRef = toArray(tensor);
  af_get_data_ref_count(&refCount, aNoRef.get());
  ASSERT_EQ(refCount, 2);
}

TEST(TensorBaseTest, BinaryOperators) {
  // TODO:fl::Tensor {testing} expand this test
  // Ensure that some binary operators work properly.
  auto a = toTensor<ArrayFireTensor>(af::constant(1, {2, 2}));
  auto b = toTensor<ArrayFireTensor>(af::constant(2, {2, 2}));
  auto c = toTensor<ArrayFireTensor>(af::constant(3, {2, 2}));

  ASSERT_TRUE(allClose(toArray(a == b), (toArray(a) == toArray(b))));
  ASSERT_TRUE(allClose((a == b), eq(a, b)));
  ASSERT_TRUE(allClose((a + b), c));
  ASSERT_TRUE(allClose((a + b), add(a, b)));
}

TEST(TensorBaseTest, full) {
  // TODO: expand with fixtures for each type
  auto a = fl::full({3, 4}, 3.);
  ASSERT_EQ(a.shape(), Shape({3, 4}));
  ASSERT_EQ(a.type(), fl::dtype::f32);
  ASSERT_TRUE(allClose(toArray(a), af::constant(3., {3, 4})));

  auto b = fl::full({1, 1, 5, 4}, 4.5);
  ASSERT_EQ(b.shape(), Shape({1, 1, 5, 4}));
  ASSERT_EQ(b.type(), fl::dtype::f32);
  ASSERT_TRUE(allClose(toArray(b), af::constant(4.5, {1, 1, 5, 4})));
}

TEST(TensorBaseTest, identity) {
  auto a = fl::identity(6);
  ASSERT_EQ(a.shape(), Shape({6, 6}));
  ASSERT_EQ(a.type(), fl::dtype::f32);
  ASSERT_TRUE(allClose(toArray(a), af::identity({6, 6})));

  ASSERT_EQ(fl::identity(6, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(TensorBaseTest, randn) {
  int s = 30;
  auto a = fl::randn({s, s});
  ASSERT_EQ(a.shape(), Shape({s, s}));
  ASSERT_EQ(a.type(), fl::dtype::f32);
  ASSERT_TRUE(af::allTrue<bool>(
      af::abs(af::mean(af::moddims(toArray(a), s * s, 1, 1, 1))) < 2));

  ASSERT_EQ(fl::randn({1}, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(TensorBaseTest, rand) {
  int s = 30;
  auto a = fl::rand({s, s});
  ASSERT_EQ(a.shape(), Shape({s, s}));
  ASSERT_EQ(a.type(), fl::dtype::f32);
  ASSERT_TRUE(af::allTrue<bool>(toArray(a) <= 1));
  ASSERT_TRUE(af::allTrue<bool>(toArray(a) >= 0));

  ASSERT_EQ(fl::rand({1}, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(TensorBaseTest, minumum) {
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

TEST(TensorBaseTest, amin) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(fl::amin<float>(a), af::min<float>(toArray(a)));
  ASSERT_TRUE(allClose(toArray(fl::amin(a, {0})), af::min(toArray(a), 0)));
}

TEST(TensorBaseTest, amax) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(fl::amax<float>(a), af::max<float>(toArray(a)));
  ASSERT_TRUE(allClose(toArray(fl::amax(a, {0})), af::max(toArray(a), 0)));
}

TEST(TensorBaseTest, sum) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(fl::sum<float>(a), af::sum<float>(toArray(a)));
  ASSERT_TRUE(allClose(toArray(fl::sum(a, {0})), af::sum(toArray(a), 0)));
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

TEST(TensorBaseTest, exp) {
  auto in = fl::full({3, 3}, 4.f);
  ASSERT_TRUE(allClose(toArray(fl::exp(in)), af::exp(toArray(in))));
}

TEST(TensorBaseTest, log) {
  auto in = fl::full({3, 3}, 2.f);
  ASSERT_TRUE(allClose(toArray(fl::log(in)), af::log(toArray(in))));
}

TEST(TensorBaseTest, log1p) {
  auto in = fl::rand({3, 3});
  ASSERT_TRUE(allClose(fl::log1p(in), fl::log(1 + in)));
}

TEST(TensorBaseTest, sin) {
  auto in = fl::rand({3, 3});
  ASSERT_TRUE(allClose(toArray(fl::sin(in)), af::sin(toArray(in))));
}

TEST(TensorBaseTest, cos) {
  auto in = fl::rand({3, 3});
  ASSERT_TRUE(allClose(toArray(fl::cos(in)), af::cos(toArray(in))));
}

TEST(TensorBaseTest, sqrt) {
  auto in = fl::full({3, 3}, 4.f);
  ASSERT_TRUE(allClose(fl::sqrt(in), in / 2));
}

TEST(TensorBaseTest, tanh) {
  auto in = fl::rand({3, 3});
  ASSERT_TRUE(allClose(toArray(fl::tanh(in)), af::tanh(toArray(in))));
}

TEST(TensorBaseTest, absolute) {
  float val = -3.1;
  ASSERT_TRUE(allClose(fl::abs(fl::full({3, 3}, val)), fl::full({3, 3}, -val)));
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

TEST(TensorBaseTest, power) {
  auto a = fl::full({3, 3}, 2.);
  auto b = fl::full({3, 3}, 2.);
  ASSERT_TRUE(allClose(fl::power(a, b), a * b));
}

TEST(TensorBaseTest, mean) {
  auto a = fl::rand({3, 50});
  ASSERT_EQ(fl::mean<float>(a), af::mean<float>(toArray(a)));
  ASSERT_TRUE(allClose(toArray(fl::mean(a, {0})), af::mean(toArray(a), 0)));
}

TEST(TensorBaseTest, var) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(fl::var<float>(a), af::var<float>(toArray(a)));
  ASSERT_TRUE(allClose(toArray(fl::var(a, {0})), af::var(toArray(a), 0)));
  ASSERT_TRUE(
      allClose(toArray(fl::var(a, {1}, false)), af::var(toArray(a), false, 1)));
  // Make sure multidimension matches computing for all
  ASSERT_FLOAT_EQ(
      toArray(fl::var(a, {0, 1}, false)).scalar<float>(),
      af::var<float>(toArray(a)));
  ASSERT_FLOAT_EQ(
      toArray(fl::var(a, {0, 1}, true)).scalar<float>(),
      af::var<float>(toArray(a), true));
}

TEST(TensorBaseTest, norm) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(fl::norm(a), af::norm(toArray(a)));
}
