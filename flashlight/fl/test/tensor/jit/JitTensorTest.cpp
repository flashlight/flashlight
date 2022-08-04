/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/jit/JitTensor.h"
#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/Utils.h"

using namespace fl;

TEST(JitTensorTest, constructor) {
  fl::setDefaultTensorType<ArrayFireTensor>();
  const auto data = rand(Shape({2, 2}), dtype::f32);
  fl::setDefaultTensorType<JitTensor<ArrayFireTensor>>();
  const Tensor tensor =
      Tensor::fromBuffer(data.shape(), data.host<float>(), Location::Host);
  const auto& jitTensor = toJitTensorBase(tensor);
  const auto node = jitTensor.node();
  ASSERT_EQ(node->inputs(), NodeList({}));
  ASSERT_EQ(node->getRefCount(), 1);
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_TRUE(node->isValue());
  ASSERT_TRUE(allClose(data, node->getResult().value()));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
