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
    const fl::Shape& strides) {
  ASSERT_EQ(tensor.shape(), shape);
  ASSERT_EQ(tensor.backendType(), fl::TensorBackendType::OneDnn);
  ASSERT_EQ(tensor.type(), type);
  ASSERT_EQ(tensor.location(), location);
  ASSERT_FALSE(tensor.isSparse());
  ASSERT_TRUE(tensor.isContiguous());
  ASSERT_EQ(tensor.strides(), strides);
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
}

TEST(OneDnnTensorTest, rawDataConstructor0D) {
  const fl::Shape shape{};
  const std::vector<int> data{42};
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1};
  OneDnnTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides);
}

TEST(OneDnnTensorTest, rawDataConstructor1D) {
  const fl::Shape shape{2};
  const std::vector<int> data{1, 2};
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1};
  OneDnnTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides);
}

TEST(OneDnnTensorTest, rawDataConstructor2D) {
  const fl::Shape shape{2, 3};
  const std::vector<int> data{2, 3, 4, 5, 6, 7};
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1, 2};
  OneDnnTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides);
}

TEST(OneDnnTensorTest, rawDataConstructor3D) {
  const fl::Shape shape{2, 3, 4};
  const std::vector<int> data(shape.elements());
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1, 2, 6};
  OneDnnTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides);
}

TEST(OneDnnTensorTest, rawDataConstructor4D) {
  const fl::Shape shape{2, 3, 4, 5};
  const std::vector<int> data(shape.elements());
  const fl::dtype type = fl::dtype::s32;
  const fl::Location location = fl::Location::Host;
  const fl::Shape strides{1, 2, 6, 24};
  OneDnnTensor tensor(shape, type, data.data(), location);
  assertRawDataConstructor(tensor, shape, type, location, strides);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
