/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <stdexcept>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/TensorExtension.h"

using namespace ::testing;
using namespace fl;

// Extension interface
class TestTensorExtension : public TensorExtension<TestTensorExtension> {
 public:
  static constexpr TensorExtensionType extensionType =
      TensorExtensionType::Generic;

  TestTensorExtension() = default;
  virtual ~TestTensorExtension() = default;

  virtual Tensor testExtensionFunc(const Tensor& tensor) = 0;
};

// Specific extension implementation
class TestArrayFireTensorExtension : public TestTensorExtension {
 public:
  Tensor testExtensionFunc(const Tensor& tensor) override {
    return tensor + 1;
  }

  bool isDataTypeSupported(const fl::dtype&) const override {
    return true;
  }
};

// Op in API
Tensor testExtensionFunc(const Tensor& tensor) {
  return tensor.backend().getExtension<TestTensorExtension>().testExtensionFunc(
      tensor);
}

FL_REGISTER_TENSOR_EXTENSION(TestArrayFireTensorExtension, ArrayFire);

TEST(TensorExtensionTest, TestExtension) {
  auto a = fl::rand({4, 5, 6});

  // TODO: this test only works with the ArrayFire backend - gate accordingly
  if (Tensor().backendType() != TensorBackendType::ArrayFire) {
    GTEST_SKIP() << "Flashlight not built with ArrayFire backend.";
  }

  // TODO: add a fixture to check with available backends
  ASSERT_TRUE(::fl::registerTensorExtension<TestArrayFireTensorExtension>(
      TensorBackendType::ArrayFire));

  ASSERT_TRUE(allClose(testExtensionFunc(a), a + 1));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
