/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
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
  auto rhsStr = oneDnnRhs.toString();
  ASSERT_TRUE(oneDnnLhs.equals(std::move(oneDnnRhs)))
   << "lhs:\n" << oneDnnLhs.toString()
   << "rhs:\n" << rhsStr;
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
  auto tFloat = fl::Tensor::fromVector(shape, std::vector<float>(shape.elements(), 40.7));
  auto tInt = fl::Tensor::fromVector(shape, std::vector<int>(shape.elements(), 42));
  assertOneDnnTensorEq(tFloat, fl::full(shape, 40.7, fl::dtype::f32));
  assertOneDnnTensorEq(tInt, fl::full(shape, 42, fl::dtype::s32));
}

TEST(OneDnnTensorTest, astype) {
  // some casts like u8 -> f16, aren't supported in OneDNN on certain platforms.
  auto tInt = fl::full({2, 2, 2}, 40, fl::dtype::s32);
  auto tFloat = fl::full({2, 2, 2}, 40.0f, fl::dtype::f32);
  assertOneDnnTensorEq(tInt, tFloat.astype(fl::dtype::s32));
  assertOneDnnTensorEq(tFloat, tInt.astype(fl::dtype::f32));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  fl::setDefaultTensorType<OneDnnTensor>();
  return RUN_ALL_TESTS();
}
