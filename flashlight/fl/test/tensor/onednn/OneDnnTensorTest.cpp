/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnBackend.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnTensor.h"

using fl::OneDnnTensor;

namespace {

void assertRawDataConstructor(
    OneDnnTensor& tensor,
    const fl::Shape& shape,
    const fl::dtype type,
    const fl::Location location,
    const fl::Shape& strides,
    const int scalar) {
  ASSERT_EQ(tensor.shape(), shape);
  ASSERT_EQ(tensor.backendType(), fl::TensorBackendType::OneDnn);
  ASSERT_EQ(tensor.type(), type);
  ASSERT_EQ(tensor.location(), location);
  ASSERT_FALSE(tensor.isSparse());
  ASSERT_TRUE(tensor.isContiguous());
  ASSERT_EQ(tensor.strides(), strides);
  int scalarVar = 0;
  ASSERT_NO_THROW(tensor.scalar(&scalarVar));
  ASSERT_EQ(scalarVar, scalar);
}

template <typename T>
static OneDnnTensor fromVector(const fl::Shape& s, const std::vector<T>& v) {
  assert(s.elements() == v.size());
  return OneDnnTensor(
      s, fl::dtype_traits<T>::fl_type, v.data(), fl::Location::Host);
}

void assertOneDnnTensorEq(fl::Tensor& lhs, fl::Tensor&& rhs) {
  auto& oneDnnLhs = lhs.getAdapter<OneDnnTensor>();
  auto& oneDnnRhs = rhs.getAdapter<OneDnnTensor>();
  ASSERT_TRUE(oneDnnLhs.equals(std::move(oneDnnRhs)))
      << "lhs:\n"
      << oneDnnLhs.toString() << "rhs:\n"
      << oneDnnRhs.toString();
}

void assertOneDnnTensorEq(fl::Tensor&& lhs, fl::Tensor&& rhs) {
  // we know it's safe, because the reference parameter won't be stored anywhere
  fl::Tensor& lhsRef = lhs;
  assertOneDnnTensorEq(lhsRef, std::move(rhs));
}

// usually we just need to specify output type, like mapFunc<float>(...);
template <typename Out, typename In>
std::vector<Out> mapFunc(const std::vector<In>& inputs, Out (*func)(In)) {
  return mapFunc<Out>(inputs, [func](In x) { return func(x); });
}

template <typename Out, typename In, typename Func>
std::vector<Out> mapFunc(const std::vector<In>& inputs, Func func) {
  std::vector<Out> outputs;
  for (const auto& input : inputs) {
    outputs.emplace_back(func(input));
  }
  return outputs;
}

} // namespace

TEST(OneDnnTensorTest, emptyConstructor) {
  OneDnnTensor tensor;
  ASSERT_EQ(tensor.shape().elements(), 0);
  ASSERT_EQ(tensor.backendType(), fl::TensorBackendType::OneDnn);
  ASSERT_EQ(tensor.type(), fl::dtype::f32);
  ASSERT_EQ(tensor.location(), fl::Location::Host);
  ASSERT_FALSE(tensor.isSparse());
  ASSERT_TRUE(tensor.isContiguous());
  ASSERT_EQ(tensor.strides(), fl::Shape({1}));
  int scalar = 0;
  ASSERT_THROW(tensor.scalar(&scalar), std::invalid_argument);
  ASSERT_EQ(tensor.toString(), "[]\n");
}

TEST(OneDnnTensorTest, rawDataConstructor0D) {
  const fl::Shape shape{};
  const std::vector<int> data{42};
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1};
  OneDnnTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides, data[0]);
}

TEST(OneDnnTensorTest, rawDataConstructor1D) {
  const fl::Shape shape{2};
  const std::vector<int> data{1, 2};
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1};
  OneDnnTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides, data[0]);
}

TEST(OneDnnTensorTest, rawDataConstructor2D) {
  const fl::Shape shape{2, 3};
  const std::vector<int> data{2, 3, 4, 5, 6, 7};
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1, 2};
  OneDnnTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides, data[0]);
}

TEST(OneDnnTensorTest, rawDataConstructor3D) {
  const fl::Shape shape{2, 3, 4};
  const std::vector<int> data(shape.elements());
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1, 2, 6};
  OneDnnTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides, data[0]);
}

TEST(OneDnnTensorTest, rawDataConstructor4D) {
  const fl::Shape shape{2, 3, 4, 5};
  const std::vector<int> data(shape.elements());
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1, 2, 6, 24};
  OneDnnTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides, data[0]);
}

TEST(OneDnnTensorTest, toString) {
  // NOTE using `char` to make sure we don't print out bytes as ascii chars
  // empty
  ASSERT_EQ(fromVector<char>({0}, {}).toString(), "[]\n");
  ASSERT_EQ(fromVector<char>({0, 0}, {}).toString(), "[]\n");

  // 1D
  ASSERT_EQ(fromVector<char>({}, {0}).toString(), "[0]\n");

  // 2D
  ASSERT_EQ(
      fromVector<char>({4}, {0, 1, 2, 3}).toString(),
      "[0,\n"
      " 1,\n"
      " 2,\n"
      " 3]\n");

  // 2D
  ASSERT_EQ(
      fromVector<char>({3, 2}, {0, 1, 2, 3, 4, 5}).toString(),
      "[[0, 3],\n"
      " [1, 4],\n"
      " [2, 5]]\n");

  // 3D
  ASSERT_EQ(
      fromVector<char>({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
          .toString(),
      "[[[0, 2, 4],\n"
      "  [1, 3, 5]],\n"
      " [[6, 8, 10],\n"
      "  [7, 9, 11]]]\n");
}

TEST(OneDnnTensorTest, equals) {
  const fl::Shape shape{2, 3, 4};
  const fl::Location location = fl::Location::Host;

  {
    std::vector<int> data(shape.elements());
    const fl::dtype type = fl::dtype::s32;
    std::generate(data.begin(), data.end(), std::rand);
    OneDnnTensor t1 = OneDnnTensor(shape, type, data.data(), location);
    ASSERT_TRUE(t1.equals(OneDnnTensor(shape, type, data.data(), location)));
    data[0]++;
    ASSERT_FALSE(t1.equals(OneDnnTensor(shape, type, data.data(), location)));
  }

  {
    std::vector<float> data(shape.elements());
    const fl::dtype type = fl::dtype::f32;
    std::generate(data.begin(), data.end(), std::rand);
    OneDnnTensor t1 = OneDnnTensor(shape, type, data.data(), location);
    ASSERT_TRUE(t1.equals(OneDnnTensor(shape, type, data.data(), location)));
    data[0] = (data[0] + 1) * (data[0] + 1); // make sure it's different enough
    ASSERT_FALSE(t1.equals(OneDnnTensor(shape, type, data.data(), location)));
  }

  {
    std::vector<char> data(shape.elements());
    const fl::dtype type = fl::dtype::u8;
    std::generate(data.begin(), data.end(), std::rand);
    OneDnnTensor t1 = OneDnnTensor(shape, type, data.data(), location);
    ASSERT_TRUE(t1.equals(OneDnnTensor(shape, type, data.data(), location)));
    data[0]++;
    ASSERT_FALSE(t1.equals(OneDnnTensor(shape, type, data.data(), location)));
  }
}

TEST(OneDnnTensorTest, transpose) {
  // [[[0, 2, 4],
  //   [1, 3, 5]],
  //  [[6, 8, 10],
  //   [7, 9, 11]]]
  auto t1 = fl::Tensor::fromVector<int>(
      {2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  assertOneDnnTensorEq(t1, fl::transpose(fl::transpose(t1)));

  // [[[0, 2, 4],
  //   [6, 8, 10]],
  //  [[1, 3, 5],
  //   [7, 9, 11]]]
  auto t2 = fl::Tensor::fromVector<int>(
      {2, 3, 2}, {0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11});
  assertOneDnnTensorEq(t2, fl::transpose(t1));

  // [[[0, 1],
  //   [6, 7]],
  //  [[2, 3],
  //   [8, 9]],
  //  [[4, 5],
  //   [10, 11]]]
  auto t3 = fl::Tensor::fromVector<int>(
      {2, 2, 3}, {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11});
  ASSERT_EQ(fl::Shape({2, 2, 3}), fl::transpose(t1, {2, 0, 1}).shape());
  assertOneDnnTensorEq(t3, fl::transpose(t1, {2, 0, 1}));
}

TEST(OneDnnTensorTest, full) {
  const fl::Shape shape{2, 2, 2};
  assertOneDnnTensorEq(
      fl::Tensor::fromVector(shape, std::vector<float>(shape.elements(), 40.7)),
      fl::full(shape, 40.7, fl::dtype::f32));
  assertOneDnnTensorEq(
      fl::Tensor::fromVector(shape, std::vector<int>(shape.elements(), 42)),
      fl::full(shape, 42, fl::dtype::s32));
}

TEST(OneDnnTensorTest, astype) {
  // some casts like u8 -> f16, aren't supported in OneDNN on certain platforms.
  auto tInt = fl::full({2, 2, 2}, 40, fl::dtype::s32);
  auto tFloat = fl::full({2, 2, 2}, 40.0f, fl::dtype::f32);
  assertOneDnnTensorEq(tInt, tFloat.astype(fl::dtype::s32));
  assertOneDnnTensorEq(tFloat, tInt.astype(fl::dtype::f32));
}

TEST(OneDnnTensorTest, host) {
  const std::vector<int> data{0, 1, 2, 3};
  std::vector<int> temp(4, 0);
  auto t = fromVector({2, 2}, data);

  // check temp.data() is propagated with tensor data
  ASSERT_NE(data, temp);
  t.host(temp.data());
  ASSERT_EQ(data, temp);

  // check temp.data() isn't "mirrored" to tensor data
  temp.data()[0] = 42;
  ASSERT_TRUE(t.equals(fromVector({2, 2}, data)));
}

TEST(OneDnnTensorTest, device) {
  auto t = fromVector<int>({2, 2}, {0, 1, 2, 3});
  void* devicePtr = nullptr;

  t.device(&devicePtr);
  ASSERT_NE(devicePtr, nullptr);
  ASSERT_TRUE(t.isLocked());
  t.unlock();
}

TEST(OneDnnTensorTest, arithmetics) {
  auto t1 = fl::Tensor::fromVector<float>({2, 2}, {0, 1, 2, 3});
  auto t2 = fl::Tensor::fromVector<int>({2, 2}, {1, 2, 3, 4});
  auto t3 = fl::Tensor::fromVector<int>({2, 2}, {3, 5, 7, 9});

  assertOneDnnTensorEq( // implicit casting
      t1 + t2,
      fl::Tensor::fromVector<float>({2, 2}, {1, 3, 5, 7}));

  // literal with casting
  auto t = fl::Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  double oneDouble = 1.0f;
  double oneInt = 1;
  assertOneDnnTensorEq(t, t1 + oneDouble);
  assertOneDnnTensorEq(t, oneDouble + t1);
  assertOneDnnTensorEq(t, t1 + oneInt);
  assertOneDnnTensorEq(t, oneInt + t1);

  assertOneDnnTensorEq(
      t3 - t2, fl::Tensor::fromVector<int>({2, 2}, {2, 3, 4, 5}));

  assertOneDnnTensorEq(
      t1 * t2, fl::Tensor::fromVector<float>({2, 2}, {0, 2, 6, 12}));

  assertOneDnnTensorEq(
      t3 / t2, fl::Tensor::fromVector<int>({2, 2}, {3, 2, 2, 2}));
}

TEST(OneDnnTensorTest, comparison) {
  auto t1 = fl::Tensor::fromVector<float>({2, 2}, {0, 1, 2, 3});
  auto t2 = fl::Tensor::fromVector<float>({2, 2}, {0, 2, 2, 4});

  assertOneDnnTensorEq(
      t1 == t2, fl::Tensor::fromVector<char>({2, 2}, {1, 0, 1, 0}));

  // literals with casting
  auto t = fl::Tensor::fromVector<char>({2, 2}, {0, 1, 1, 0});
  double twoDouble = 2.0f;
  double twoInt = 2;
  assertOneDnnTensorEq(t, t2 == twoDouble);
  assertOneDnnTensorEq(t, twoDouble == t2);
  assertOneDnnTensorEq(t, t2 == twoInt);
  assertOneDnnTensorEq(t, twoInt == t2);

  assertOneDnnTensorEq(
      t1 != t2, fl::Tensor::fromVector<char>({2, 2}, {0, 1, 0, 1}));

  assertOneDnnTensorEq(
      t1 < t2, fl::Tensor::fromVector<char>({2, 2}, {0, 1, 0, 1}));

  assertOneDnnTensorEq(
      t1 <= t2, fl::Tensor::fromVector<char>({2, 2}, {1, 1, 1, 1}));

  assertOneDnnTensorEq(
      t1 > t2, fl::Tensor::fromVector<char>({2, 2}, {0, 0, 0, 0}));

  assertOneDnnTensorEq(
      t1 >= t2, fl::Tensor::fromVector<char>({2, 2}, {1, 0, 1, 0}));
}

TEST(OneDnnTensorTest, minimumMaximum) {
  auto t1 = fl::Tensor::fromVector<float>({2, 2}, {0, 1, 2, 3});
  auto t2 = fl::Tensor::fromVector<float>({2, 2}, {0, 2, 2, 4});

  assertOneDnnTensorEq(t2, fl::maximum(t1, t2));
  assertOneDnnTensorEq(
      fl::maximum(t1, 2), fl::Tensor::fromVector<float>({2, 2}, {2, 2, 2, 3}));
  assertOneDnnTensorEq(
      fl::maximum(2, t1), fl::Tensor::fromVector<float>({2, 2}, {2, 2, 2, 3}));

  assertOneDnnTensorEq(t1, fl::minimum(t1, t2));
  assertOneDnnTensorEq(
      fl::minimum(t1, 1), fl::Tensor::fromVector<float>({2, 2}, {0, 1, 1, 1}));
  assertOneDnnTensorEq(
      fl::minimum(1, t1), fl::Tensor::fromVector<float>({2, 2}, {0, 1, 1, 1}));
}

TEST(OneDnnTensorTest, logicalBinops) {
  auto t1 = fl::Tensor::fromVector<char>({2, 2}, {0, 3, 0, 5});
  auto t2 = fl::Tensor::fromVector<int>({2, 2}, {0, 2, -1, 0});
  auto t3 = fl::Tensor::fromVector<int>({2}, {0, 2, -1, 0});
  auto t4 = fl::Tensor::fromVector<int>({2, 1}, {0, 2, -1, 0});

  assertOneDnnTensorEq(
      t1 && t2, fl::Tensor::fromVector<char>({2, 2}, {0, 1, 0, 0}));
  assertOneDnnTensorEq(
      t1 || t2, fl::Tensor::fromVector<char>({2, 2}, {0, 1, 1, 1}));
  ASSERT_THROW(t1 && t3, std::invalid_argument);
  ASSERT_THROW(t4 || t1, std::invalid_argument);
}

TEST(OneDnnTensorTest, assign) {
  const auto type = fl::dtype::f32;
  auto t1 = fl::full({2, 2}, 40.7, type);
  auto t2 = fl::full({2, 2}, 23, type);

  // ensure it's not a shallow copy
  t2 = t1;
  t1(0) = fl::full({2}, 0, type);
  assertOneDnnTensorEq(
      t1, fl::Tensor::fromVector<float>({2, 2}, {0, 40.7, 0, 40.7}, type));
  assertOneDnnTensorEq(t2, fl::full({2, 2}, 40.7, type));
}

TEST(OneDnnTensorTest, copy) {
  const auto type = fl::dtype::f32;
  auto t1 = fl::full({2, 2}, 40.7, type);
  assertOneDnnTensorEq(t1, t1.copy());

  // ensure it's not a shallow copy
  auto t2 = t1.copy();
  t1(0) = fl::full({2}, 0, type);
  assertOneDnnTensorEq(
      t1, fl::Tensor::fromVector<float>({2, 2}, {0, 40.7, 0, 40.7}, type));
  assertOneDnnTensorEq(t2, fl::full({2, 2}, 40.7, type));
}

TEST(OneDnnTensorTest, rand) {
  fl::Shape shape{2, 2, 2};
  auto type = fl::dtype::f32;
  auto t1 = fl::OneDnnBackend::getInstance().randn(shape, type);
  ASSERT_EQ(t1.shape(), shape);
  ASSERT_EQ(t1.type(), type);
}

TEST(OneDnnTensorTest, eltwiseCompute) {
  std::vector<float> t1Data = {1, 2, 3, 4};
  std::vector<float> t2Data = {0, 2, 2, 0};
  auto t1 = fl::Tensor::fromVector({2, 2}, t1Data);
  auto t2 = fl::Tensor::fromVector({2, 2}, t2Data);

  assertOneDnnTensorEq(
      fl::exp(t1),
      fl::Tensor::fromVector({2, 2}, mapFunc<float>(t1Data, std::exp)));

  assertOneDnnTensorEq(
      fl::log(t1),
      fl::Tensor::fromVector({2, 2}, mapFunc<float>(t1Data, std::log)));

  assertOneDnnTensorEq(
      fl::sqrt(t1),
      fl::Tensor::fromVector({2, 2}, mapFunc<float>(t1Data, std::sqrt)));

  assertOneDnnTensorEq(
      fl::tanh(t1),
      fl::Tensor::fromVector({2, 2}, mapFunc<float>(t1Data, std::tanh)));

  assertOneDnnTensorEq(
      fl::rint(fl::Tensor::fromVector<float>({2, 2}, {0.1, 1.4, 2.5, 3.9})),
      fl::Tensor::fromVector<float>({2, 2}, {0, 1, 2, 4}));

  assertOneDnnTensorEq(
      fl::absolute(fl::Tensor::fromVector<float>({2, 2}, {-2, -2.8, 3.4, 0})),
      fl::Tensor::fromVector<float>({2, 2}, {2, 2.8, 3.4, 0}));

  assertOneDnnTensorEq(
      fl::logicalNot(t2), fl::Tensor::fromVector<char>({2, 2}, {1, 0, 0, 1}));

  assertOneDnnTensorEq(
      fl::clip(
          fl::Tensor::fromVector<float>({4}, {1.4, 0.1, 2.5, 3.9}), 1.2, 2.7),
      fl::Tensor::fromVector<float>({4}, {1.4, 1.2, 2.5, 2.7}));

  auto powThree = [](auto x) { return std::pow(x, 3); };
  assertOneDnnTensorEq(
      fl::power(t1, 3),
      fl::Tensor::fromVector(t1.shape(), mapFunc<float>(t1Data, powThree)));

  assertOneDnnTensorEq(
      fl::erf(t1),
      fl::Tensor::fromVector({2, 2}, mapFunc<float>(t1Data, std::erf)));
}

TEST(OneDnnTensorTest, eltwiseLogical) {
  std::vector<float> t1Data = {
      std::numeric_limits<float>::infinity(),
      2,
      0.2f,
      -std::numeric_limits<float>::infinity(),
      -2,
      0};
  auto t1 = fl::Tensor::fromVector({2, 3}, t1Data);

  assertOneDnnTensorEq(
      fl::isinf(t1),
      fl::Tensor::fromVector(t1.shape(), mapFunc<char>(t1Data, [](auto x) {
                               return std::isinf(x);
                             })));

  auto sign = [](auto x) {
    if (x > 0) {
      return 1;
    } else if (x == 0) {
      return 0;
    } else {
      return -1;
    }
  };
  assertOneDnnTensorEq(
      fl::sign(t1),
      fl::Tensor::fromVector(t1.shape(), mapFunc<char>(t1Data, sign)));
}

TEST(OneDnnTensorTest, reduction) {
  std::vector<float> data = {-1, 0, 33, 0};
  auto t0 = fl::Tensor::fromVector({2, 2}, data);
  auto t1 = fl::full({2, 3, 4}, 42, fl::dtype::f32);

  assertOneDnnTensorEq(fl::sum(t1, {1}), fl::full({2, 4}, 126, fl::dtype::f32));
  assertOneDnnTensorEq(
      fl::amax(t0, {1}), fl::Tensor::fromVector<float>({2}, {33, 0}));
  assertOneDnnTensorEq(fl::amin(t0), fl::Tensor::fromVector<float>({}, {-1}));
  assertOneDnnTensorEq(
      fl::countNonzero(t0), fl::Tensor::fromVector<int>({}, {2}));
  assertOneDnnTensorEq(
      fl::sum(t1, {}, true),
      fl::full({1, 1, 1}, 42 * 2 * 3 * 4, fl::dtype::f32));
}

TEST(OneDnnTensorTest, matmul) {
  using MP = fl::MatrixProperty;
  auto& backend = fl::OneDnnBackend::getInstance();
  // 1 2 3  X  2 5  =  20 38
  // 4 5 6     3 6     47 92
  //           4 7
  auto t1 = fl::Tensor::fromVector<float>({2, 3}, {1, 4, 2, 5, 3, 6});
  auto t2 = fl::Tensor::fromVector<float>({3, 2}, {2, 3, 4, 5, 6, 7});
  auto res1 = backend.matmul(t1, t2, MP::None, MP::None);
  assertOneDnnTensorEq(
      res1, fl::Tensor::fromVector<float>({2, 2}, {20, 47, 38, 92}));

  // 1 4  X  2 3 4  = 22 27 32
  // 2 5     5 6 7    29 36 43
  // 3 6              36 45 54
  auto res2 = backend.matmul(t1, t2, MP::Transpose, MP::Transpose);
  assertOneDnnTensorEq(
      res2,
      fl::Tensor::fromVector<float>(
          {3, 3}, {22, 29, 36, 27, 36, 45, 32, 43, 54}));
}

TEST(OneDnnTensorTest, matmulShapes) {
  using MP = fl::MatrixProperty;
  auto& backend = fl::OneDnnBackend::getInstance();
  struct Input {
    fl::Shape lhsShape;
    fl::Shape rhsShape;
    fl::MatrixProperty lhsMP;
    fl::MatrixProperty rhsMP;
    std::optional<fl::Shape> dstShape;
  };
  const std::vector<Input> inputs = {
      // scalar/vector
      {{}, {}, MP::None, MP::None, {{1}}},
      {{2}, {}, MP::None, MP::None, std::nullopt},
      {{}, {3}, MP::None, MP::None, std::nullopt},
      {{4}, {4}, MP::None, MP::None, {{1}}},
      {{2}, {3}, MP::None, MP::None, std::nullopt},
      {{2}, {2, 3}, MP::None, MP::None, {{3}}},
      {{2, 3}, {3}, MP::None, MP::None, {{2}}},
      // matrix
      {{3, 2}, {2, 3}, MP::None, MP::None, {{3, 3}}},
      {{3, 2}, {2, 3}, MP::Transpose, MP::Transpose, {{2, 2}}},
      {{2, 3}, {2, 3}, MP::Transpose, MP::None, {{3, 3}}},
      {{2, 3}, {2, 3}, MP::None, MP::Transpose, {{2, 2}}},
      {{2, 3}, {2, 3}, MP::None, MP::Transpose, {{2, 2}}},
      // batch matrix
      {{2, 3, 42}, {2, 3, 42}, MP::None, MP::Transpose, {{2, 2, 42}}},
      {{2, 3, 41}, {2, 3, 42}, MP::None, MP::Transpose, std::nullopt},
      // TODO support broadcast
      {{2, 3, 1}, {2, 3, 42}, MP::None, MP::Transpose, std::nullopt},
      {{2, 3}, {2, 3, 42}, MP::None, MP::Transpose, std::nullopt},
  };
  for (auto& input : inputs) {
    const auto lhs = backend.rand(input.lhsShape, fl::dtype::f32);
    const auto rhs = backend.rand(input.rhsShape, fl::dtype::f32);
    if (input.dstShape.has_value()) {
      const auto res = backend.matmul(lhs, rhs, input.lhsMP, input.rhsMP);
      ASSERT_EQ(res.shape(), input.dstShape.value())
          << input.lhsShape << " X " << input.rhsShape;
    } else {
      ASSERT_THROW(
          backend.matmul(lhs, rhs, input.lhsMP, input.rhsMP),
          std::invalid_argument);
    }
  }
}

TEST(OneDnnTensorTest, max) {
  using fl::Shape;
  using fl::Tensor;
  // 1 4 5
  // 2 3 6
  Tensor in = Tensor::fromVector<float>({2, 3}, {1, 2, 4, 3, 5, 6});
  Tensor values, indices;

  fl::max(values, indices, in, 0);
  assertOneDnnTensorEq(indices, Tensor::fromVector<int>({3}, {1, 0, 1}));
  assertOneDnnTensorEq(values, Tensor::fromVector<float>({3}, {2, 4, 6}));

  fl::max(values, indices, in, 1);
  assertOneDnnTensorEq(indices, Tensor::fromVector<int>({2}, {2, 2}));
  assertOneDnnTensorEq(values, Tensor::fromVector<float>({2}, {5, 6}));

  fl::max(values, indices, in, 0, /* keepDims = */ true);
  assertOneDnnTensorEq(indices, Tensor::fromVector<int>({1, 3}, {1, 0, 1}));
  assertOneDnnTensorEq(values, Tensor::fromVector<float>({1, 3}, {2, 4, 6}));

  fl::max(values, indices, in, 1, /* keepDims = */ true);
  assertOneDnnTensorEq(indices, Tensor::fromVector<int>({2, 1}, {2, 2}));
  assertOneDnnTensorEq(values, Tensor::fromVector<float>({2, 1}, {5, 6}));
}

TEST(OneDnnTensorTest, min) {
  using fl::Shape;
  using fl::Tensor;
  // 1 4 5
  // 2 3 6
  Tensor in = Tensor::fromVector<float>({2, 3}, {1, 2, 4, 3, 5, 6});
  Tensor values, indices;

  fl::min(values, indices, in, 0);
  assertOneDnnTensorEq(indices, Tensor::fromVector<int>({3}, {0, 1, 0}));
  assertOneDnnTensorEq(values, Tensor::fromVector<float>({3}, {1, 3, 5}));

  fl::min(values, indices, in, 1);
  assertOneDnnTensorEq(indices, Tensor::fromVector<int>({2}, {0, 0}));
  assertOneDnnTensorEq(values, Tensor::fromVector<float>({2}, {1, 2}));

  fl::min(values, indices, in, 0, /* keepDims = */ true);
  assertOneDnnTensorEq(indices, Tensor::fromVector<int>({1, 3}, {0, 1, 0}));
  assertOneDnnTensorEq(values, Tensor::fromVector<float>({1, 3}, {1, 3, 5}));

  fl::min(values, indices, in, 1, /* keepDims = */ true);
  assertOneDnnTensorEq(indices, Tensor::fromVector<int>({2, 1}, {0, 0}));
  assertOneDnnTensorEq(values, Tensor::fromVector<float>({2, 1}, {1, 2}));
}

TEST(OneDnnTensorTest, reshape) {
  auto a = fl::full({4, 4}, 3.);
  auto b = fl::reshape(a, fl::Shape({8, 2}));
  assertOneDnnTensorEq(b, fl::full({8, 2}, 3.));
  assertOneDnnTensorEq(a, fl::reshape(b, {4, 4}));
  assertOneDnnTensorEq(fl::reshape(a, fl::Shape({16})), fl::full({16}, 3.));

  ASSERT_THROW(fl::reshape(a, {}), std::invalid_argument);
  ASSERT_THROW(fl::reshape(a, {4}), std::invalid_argument);
  ASSERT_THROW(fl::reshape(a, {4, 5}), std::invalid_argument);
  ASSERT_NO_THROW(fl::reshape(a, {4, 4, 1}));
  ASSERT_NO_THROW(fl::reshape(a, {1, 4, 4}));
  ASSERT_NO_THROW(fl::reshape(a, {4, 1, 4}));
  ASSERT_NO_THROW(fl::reshape(a, {1, 4, 4, 1}));
}

TEST(OneDnnTensorTest, index) {
  auto a = fl::Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  // indexing for read
  assertOneDnnTensorEq(a(0), fl::Tensor::fromVector<float>({2}, {1, 3}));
  assertOneDnnTensorEq(a(0) + a(1), fl::Tensor::fromVector<float>({2}, {3, 7}));

  // indexing for write
  a(0) = fl::Tensor::fromVector<float>({2}, {0, 1});
  assertOneDnnTensorEq(a, fl::Tensor::fromVector<float>({2, 2}, {0, 2, 1, 4}));

  // indexing composability
  a(0)(1) = fl::Tensor::fromVector<float>({}, {42});
  assertOneDnnTensorEq(a, fl::Tensor::fromVector<float>({2, 2}, {0, 2, 42, 4}));
}

TEST(OneDnnTensorTest, tile) {
  auto a = fl::Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4});
  assertOneDnnTensorEq(
      fl::tile(a, {2}),
      fl::Tensor::fromVector<float>({4, 2}, {1, 2, 1, 2, 3, 4, 3, 4}));
  assertOneDnnTensorEq(
      fl::tile(a(fl::span, 1), {2}),
      fl::Tensor::fromVector<float>({4}, {3, 4, 3, 4}));
}

TEST(OneDnnTensorTest, arange) {
  // Range/step overload
  assertOneDnnTensorEq(
      fl::arange(2, 10, 2), fl::Tensor::fromVector<int>({2, 4, 6, 8}));
  assertOneDnnTensorEq(
      fl::arange(0, 6), fl::Tensor::fromVector<int>({0, 1, 2, 3, 4, 5}));

  // Shape overload
  assertOneDnnTensorEq(
      fl::arange({4}), fl::Tensor::fromVector<float>({0, 1, 2, 3}));
  assertOneDnnTensorEq(
      fl::arange({2, 2}, 1),
      fl::Tensor::fromVector<float>({2, 2}, {0, 0, 1, 1}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  fl::setDefaultTensorType<OneDnnTensor>();
  return RUN_ALL_TESTS();
}
