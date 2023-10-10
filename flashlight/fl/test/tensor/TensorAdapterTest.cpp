/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace ::testing;
using namespace fl;

TEST(TensorBaseTest, DefaultBackend) {
  Tensor t;
  ASSERT_EQ(t.backendType(), DefaultTensorType_t::tensorBackendType);
}

TEST(TensorBaseTest, ImplTypeConversion) {
  // Converting to the same type is a noop
  auto a = fl::rand({6, 8});
  auto c = a.copy();
  TensorBackendType aBackend = a.backendType();
  auto b = to<DefaultTensorType_t>(std::move(a));
  ASSERT_EQ(aBackend, b.backendType());
  ASSERT_TRUE(allClose(b, c));
}

TEST(TensorBaseTest, hasAdapter) {
  Tensor a = fromScalar(3.14, fl::dtype::f32);
  ASSERT_TRUE(a.hasAdapter());
  detail::releaseAdapterUnsafe(a);
  ASSERT_FALSE(a.hasAdapter());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
