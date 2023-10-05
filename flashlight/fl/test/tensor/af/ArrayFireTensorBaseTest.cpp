/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <exception>
#include <stdexcept>
#include <utility>

#include <af/util.h>
#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"

using namespace ::testing;
using namespace fl;

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

} // namespace

namespace fl {

TEST(ArrayFireTensorBaseTest, ArrayFireShapeInterop) {
  ASSERT_EQ(detail::afToFlDims(af::dim4(), 0), Shape({})); // scalar
  ASSERT_EQ(detail::afToFlDims(af::dim4(0), 1), Shape({0}));
  ASSERT_EQ(detail::afToFlDims(af::dim4(0, 5), 2), Shape({0, 5}));
  ASSERT_EQ(detail::afToFlDims(af::dim4(1, 0, 2), 3), Shape({1, 0, 2}));
  ASSERT_EQ(detail::afToFlDims(af::dim4(0, 1, 1, 1), 4), Shape({0, 1, 1, 1}));

  using namespace fl::detail;
  auto dimsEq = [](const af::dim4& d, const Shape& s) {
    return detail::afToFlDims(d, s.ndim()) == s;
  };

  ASSERT_TRUE(dimsEq(af::dim4(3), {3})); // not 3, 1, 1, 1
  ASSERT_TRUE(dimsEq(af::dim4(3, 2), {3, 2})); // not 3, 2, 1, 1
  ASSERT_TRUE(dimsEq(af::dim4(3, 1), {3}));
  // if explicitly specified, uses implicit 1 dim
  ASSERT_TRUE(dimsEq(af::dim4(3, 1), {3, 1}));
  ASSERT_TRUE(dimsEq(af::dim4(1, 3, 2), {1, 3, 2}));
  ASSERT_TRUE(dimsEq(af::dim4(1), {1}));
  ASSERT_TRUE(dimsEq(af::dim4(1, 1, 1), {1}));
}

} // namespace fl

namespace {

int getRefCount(const af::array& arr, bool sync = true) {
  if (sync) {
    arr.eval();
    af::sync();
  }
  int refCount = 0;
  AF_CHECK(af_get_data_ref_count(&refCount, arr.get()));
  return refCount;
}

} // namespace

TEST(ArrayFireTensorBaseTest, AfRefCountBasic) {
  // Sanity check that af::arrays moved into fl::Tensors don't have their
  // refcount inrcremented/show proper usage of refs in tensor ops
  auto q = af::constant(1, {2, 2});
  // without eval/sync, no refcount
  ASSERT_EQ(getRefCount(q, /* sync = */ false), 0);

  auto a = af::constant(1, {2, 2});
  ASSERT_EQ(getRefCount(a), 1);

  auto tensor = toTensor<ArrayFireTensor>(std::move(a), /* numDims = */ 2);
  auto& aRef = toArray(tensor);
  ASSERT_EQ(getRefCount(aRef), 1);
  // Sanity check copying bumps things
  auto aNoRef = toArray(tensor);
  ASSERT_EQ(getRefCount(aNoRef), 2);
}

TEST(ArrayFireTensorBaseTest, AfRefCountModify) {
  // Compositional operations don't increment refcount
  auto a = af::constant(1, {2, 2});
  auto b = af::constant(1, {2, 2});
  auto arrRes = a + b;
  ASSERT_EQ(getRefCount(a), 1);
  ASSERT_EQ(getRefCount(b), 1);
  // Multiple uses of the same variable doesn't push count
  auto c = af::constant(1, {2, 2});
  auto d = af::constant(1, {2, 2});
  auto arrResMult = c * c + d * d;
  ASSERT_EQ(getRefCount(c), 1);
  ASSERT_EQ(getRefCount(d), 1);

  // Same behavior with Tensors
  auto v = fl::full({2, 2}, 1);
  auto w = fl::full({2, 2}, 1);
  auto varRes = v + w;
  ASSERT_EQ(getRefCount(toArray(v)), 1);
  ASSERT_EQ(getRefCount(toArray(w)), 1);
  // Multiuse with variables
  auto y = fl::full({2, 2}, 1);
  auto z = fl::full({2, 2}, 1);
  auto varResMult = y * y + z * z;
  ASSERT_EQ(getRefCount(toArray(y)), 1);
  ASSERT_EQ(getRefCount(toArray(z)), 1);
}

TEST(ArrayFireTensorBaseTest, astypeRefcount) {
  auto t = fl::rand({5, 5});
  ASSERT_EQ(getRefCount(toArray(t)), 1);
  auto t64 = t.astype(fl::dtype::f64);
  ASSERT_EQ(getRefCount(toArray(t64)), 1);
}

TEST(ArrayFireTensorBaseTest, astypeInPlaceRefcount) {
  auto a = fl::rand({4, 4});
  ASSERT_EQ(getRefCount(toArray(a)), 1);
  a = a.astype(fl::dtype::f64);
  ASSERT_EQ(getRefCount(toArray(a)), 1);
  ASSERT_EQ(a.type(), fl::dtype::f64);
  a = a.astype(fl::dtype::f32);
  ASSERT_EQ(getRefCount(toArray(a)), 1);
}

TEST(ArrayFireTensorBaseTest, BackendInterop) {
  // TODO: test toTensorBackend here since we know we have a backend available;
  // design a test that tests with mulitple backends once available
  auto a = fl::rand({10, 12});
  ASSERT_EQ(a.backendType(), TensorBackendType::ArrayFire);
  auto b = a;
  auto t = fl::toTensorType<ArrayFireTensor>(std::move(a));
  ASSERT_EQ(t.backendType(), TensorBackendType::ArrayFire);
  ASSERT_TRUE(allClose(b, t));
}

TEST(ArrayFireTensorBaseTest, withTensorType) {
  // TODO: test with here since we know we have a backend available;
  // design a test that tests with mulitple backends once available
  Tensor t;
  fl::withTensorType<ArrayFireTensor>([&t]() {
    t = fl::full({5, 5}, 6.);
    t += 1;
  });
  ASSERT_TRUE(allClose(t, fl::full({5, 5}, 7.)));
}

TEST(ArrayFireTensorBaseTest, ArrayFireAssignmentOperators) {
  fl::Tensor a = fl::full({3, 3}, 1.);
  af::array& aArr = toArray(a);
  ASSERT_EQ(getRefCount(aArr), 1);

  auto b = a; // share the same underlying array but bump refcount
  ASSERT_EQ(getRefCount(aArr), 2);

  auto c = a.copy(); // defers deep copy to AF
  ASSERT_EQ(getRefCount(aArr), 2);

  // copy, else it'll get released below when we reassign a new tensor to b
  af::array bArr = toArray(b);
  ASSERT_EQ(getRefCount(bArr), 3); // aArr, bArr, b.arrayHandle_

  b = fl::full({4, 4}, 2.); // b points to a new array now
  ASSERT_EQ(getRefCount(bArr), 2); // aArr, bArr

  ASSERT_EQ(getRefCount(aArr), 2); // aArr, bArr
}

TEST(ArrayFireTensorBaseTest, BinaryOperators) {
  auto a =
      toTensor<ArrayFireTensor>(af::constant(1, {2, 2}), /* numDims = */ 2);
  auto b =
      toTensor<ArrayFireTensor>(af::constant(2, {2, 2}), /* numDims = */ 2);
  auto c =
      toTensor<ArrayFireTensor>(af::constant(3, {2, 2}), /* numDims = */ 2);

  ASSERT_TRUE(allClose(toArray(a == b), (toArray(a) == toArray(b))));
  ASSERT_TRUE(allClose((a == b), eq(a, b)));
  ASSERT_TRUE(allClose((a + b), c));
  ASSERT_TRUE(allClose((a + b), add(a, b)));
}

TEST(ArrayFireTensorBaseTest, full) {
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

TEST(ArrayFireTensorBaseTest, identity) {
  auto a = fl::identity(6);
  ASSERT_EQ(a.shape(), Shape({6, 6}));
  ASSERT_EQ(a.type(), fl::dtype::f32);
  ASSERT_TRUE(allClose(toArray(a), af::identity({6, 6})));

  ASSERT_EQ(fl::identity(6, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(ArrayFireTensorBaseTest, randn) {
  int s = 30;
  auto a = fl::randn({s, s});
  ASSERT_EQ(a.shape(), Shape({s, s}));
  ASSERT_EQ(a.type(), fl::dtype::f32);
  ASSERT_TRUE(af::allTrue<bool>(
      af::abs(af::mean(af::moddims(toArray(a), s * s, 1, 1, 1))) < 2));

  ASSERT_EQ(fl::randn({1}, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(ArrayFireTensorBaseTest, rand) {
  int s = 30;
  auto a = fl::rand({s, s});
  ASSERT_EQ(a.shape(), Shape({s, s}));
  ASSERT_EQ(a.type(), fl::dtype::f32);
  ASSERT_TRUE(af::allTrue<bool>(toArray(a) <= 1));
  ASSERT_TRUE(af::allTrue<bool>(toArray(a) >= 0));

  ASSERT_EQ(fl::rand({1}, fl::dtype::f64).type(), fl::dtype::f64);
}

TEST(ArrayFireTensorBaseTest, amin) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(fl::amin(a).scalar<float>(), af::min<float>(toArray(a)));
  ASSERT_TRUE(allClose(
      toArray(fl::amin(a, {0})),
      fl::detail::condenseIndices(af::min(toArray(a), 0))));
}

TEST(ArrayFireTensorBaseTest, amax) {
  auto a = fl::rand({3, 3});
  ASSERT_EQ(fl::amax(a).scalar<float>(), af::max<float>(toArray(a)));
  ASSERT_TRUE(allClose(
      toArray(fl::amax(a, {0})),
      fl::detail::condenseIndices(af::max(toArray(a), 0))));
}

TEST(ArrayFireTensorBaseTest, sum) {
  auto a = fl::rand({3, 3});
  ASSERT_NEAR(fl::sum(a).scalar<float>(), af::sum<float>(toArray(a)), 1e-5);
  ASSERT_TRUE(allClose(
      toArray(fl::sum(a, {0})),
      fl::detail::condenseIndices(af::sum(toArray(a), 0))));

  auto b = fl::rand({5, 6, 7, 8});
  ASSERT_NEAR(fl::sum(b).scalar<float>(), af::sum<float>(toArray(b)), 1e-3);
  ASSERT_TRUE(allClose(
      toArray(fl::sum(b, {1, 2})),
      fl::detail::condenseIndices(af::sum(af::sum(toArray(b), 1), 2))));
}

TEST(ArrayFireTensorBaseTest, exp) {
  auto in = fl::full({3, 3}, 4.f);
  ASSERT_TRUE(allClose(toArray(fl::exp(in)), af::exp(toArray(in))));
}

TEST(ArrayFireTensorBaseTest, log) {
  auto in = fl::full({3, 3}, 2.f);
  ASSERT_TRUE(allClose(toArray(fl::log(in)), af::log(toArray(in))));
}

TEST(ArrayFireTensorBaseTest, log1p) {
  auto in = fl::rand({3, 3});
  ASSERT_TRUE(allClose(fl::log1p(in), fl::log(1 + in)));
}

TEST(ArrayFireTensorBaseTest, sin) {
  auto in = fl::rand({3, 3});
  ASSERT_TRUE(allClose(toArray(fl::sin(in)), af::sin(toArray(in))));
}

TEST(ArrayFireTensorBaseTest, cos) {
  auto in = fl::rand({3, 3});
  ASSERT_TRUE(allClose(toArray(fl::cos(in)), af::cos(toArray(in))));
}

TEST(ArrayFireTensorBaseTest, sqrt) {
  auto in = fl::full({3, 3}, 4.f);
  ASSERT_TRUE(allClose(fl::sqrt(in), in / 2));
}

TEST(ArrayFireTensorBaseTest, tanh) {
  auto in = fl::rand({3, 3});
  ASSERT_TRUE(allClose(toArray(fl::tanh(in)), af::tanh(toArray(in))));
}

TEST(ArrayFireTensorBaseTest, absolute) {
  float val = -3.1;
  ASSERT_TRUE(allClose(fl::abs(fl::full({3, 3}, val)), fl::full({3, 3}, -val)));
}

TEST(ArrayFireTensorBaseTest, erf) {
  auto in = fl::rand({3, 3});
  ASSERT_TRUE(allClose(toArray(fl::erf(in)), af::erf(toArray(in))));
}

TEST(ArrayFireTensorBaseTest, mean) {
  auto a = fl::rand({3, 50});
  ASSERT_NEAR(fl::mean(a).scalar<float>(), af::mean<float>(toArray(a)), 1e-4);
  ASSERT_TRUE(allClose(
      toArray(fl::mean(a, {0})),
      detail::condenseIndices(af::mean(toArray(a), 0))));
}

TEST(ArrayFireTensorBaseTest, median) {
  auto a = fl::rand({3, 50});
  ASSERT_NEAR(
      fl::median(a).scalar<float>(), af::median<float>(toArray(a)), 1e-3);
  ASSERT_TRUE(allClose(
      toArray(fl::median(a, {0})),
      detail::condenseIndices(af::median(toArray(a), 0))));
}

TEST(ArrayFireTensorBaseTest, var) {
  const bool bias = false;
  af_var_bias biasMode = bias ? AF_VARIANCE_SAMPLE : AF_VARIANCE_POPULATION;
  auto a = fl::rand({3, 3});
  ASSERT_EQ(fl::var(a).scalar<float>(), af::var<float>(toArray(a), biasMode));
  ASSERT_TRUE(allClose(
      toArray(fl::var(a, {0})),
      detail::condenseIndices(af::var(toArray(a), /* mode = */ biasMode, 0))));
  ASSERT_TRUE(allClose(
      toArray(fl::var(a, {1})),
      detail::condenseIndices(af::var(toArray(a), /* mode = */ biasMode, 1))));
  // Make sure multidimension matches computing for all
  ASSERT_FLOAT_EQ(
      toArray(fl::var(a)).scalar<float>(),
      af::var<float>(toArray(a), biasMode));
  ASSERT_FLOAT_EQ(
      toArray(fl::var(a, {0, 1}, /* biased = */ true)).scalar<float>(),
      af::var<float>(toArray(a), /* mode = */ AF_VARIANCE_SAMPLE));
}

TEST(ArrayFireTensorBaseTest, std) {
  auto a = fl::rand({3, 3});
  ASSERT_TRUE(allClose(
      toArray(fl::std(a, {0}, /* keepDims = */ true)),
      af::stdev(toArray(a), AF_VARIANCE_POPULATION, 0)));
  ASSERT_TRUE(allClose(
      toArray(fl::std(a, {1}, /* keepDims = */ true)),
      af::stdev(toArray(a), AF_VARIANCE_POPULATION, 1)));
  // Make sure multidimension matches computing for all
  ASSERT_FLOAT_EQ(
      toArray(fl::std(a, {0, 1})).scalar<float>(),
      std::sqrt(af::var<float>(toArray(a), AF_VARIANCE_POPULATION)));
}

TEST(ArrayFireTensorBaseTest, norm) {
  auto a = fl::rand({3, 3});
  ASSERT_NEAR(fl::norm(a).scalar<float>(), af::norm(toArray(a)), 1e-4);
}

TEST(ArrayFireTensorBaseTest, tile) {
  auto a = fl::rand({3, 3});
  ASSERT_TRUE(allClose(
      toArray(fl::tile(a, {4, 5, 6})), af::tile(toArray(a), {4, 5, 6})));
}

TEST(ArrayFireTensorBaseTest, nonzero) {
  auto a = fl::rand({10, 10}).astype(fl::dtype::u32);
  auto nz = fl::nonzero(a);
  ASSERT_TRUE(allClose(toArray(nz), af::where(toArray(a))));
}

TEST(ArrayFireTensorBaseTest, transpose) {
  auto a = fl::rand({3, 5});
  ASSERT_THROW(fl::transpose(a, {0, 1, 2, 3, 4}), std::invalid_argument);
  ASSERT_TRUE(allClose(toArray(fl::transpose(a)), af::transpose(toArray(a))));

  auto b = fl::rand({3, 5, 4, 8});
  ASSERT_TRUE(allClose(
      toArray(fl::transpose(b, {2, 0, 1, 3})),
      af::reorder(toArray(b), 2, 0, 1, 3)));
}

TEST(ArrayFireTensorBaseTest, concatenate) {
  std::vector<fl::Tensor> tensors(11);
  ASSERT_THROW(fl::concatenate(tensors), std::invalid_argument);
}

TEST(ArrayFireTensorBaseTest, device) {
  auto a = fl::rand({5, 5});
  float* flPtr = a.device<float>();
  af::array& arr = toArray(a);
  float* afPtr = arr.device<float>();
  ASSERT_EQ(flPtr, afPtr);
  a.unlock();
  AF_CHECK(af_unlock_array(arr.get())); // safety
}

TEST(ArrayFireTensorBaseTest, defaultConstructor) {
  auto t = ArrayFireTensor();
  ASSERT_TRUE(t.getHandle().isempty());
}

TEST(ArrayFireTensorBaseTest, emptyRangeIndexing) {
  // TODO the following should all return empty tensor, but AF currently doesn't
  // have a way to represent empty range, and we are just throwing internally.
  auto t = fl::rand({5});
  ASSERT_THROW(t(fl::range(5, fl::end)).shape(), std::exception);
  ASSERT_THROW(t(fl::range(4, -1)).shape(), std::exception);
  ASSERT_THROW(t(fl::range(0, 0)).shape(), std::exception);
  ASSERT_THROW(t(fl::range(1, 1)).shape(), std::exception);
  ASSERT_THROW(t(fl::range(4, 4)).shape(), std::exception);
  ASSERT_THROW(t(fl::range(0, -5)).shape(), std::exception);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
