/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/jit/JitTensor.h"
#include "flashlight/fl/tensor/backend/jit/JitTensorBase.h"
#include "flashlight/fl/tensor/backend/jit/Utils.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"

using namespace fl;

namespace {

class JitTensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fl::setDefaultTensorType<JitTensor<ArrayFireTensor>>();
  }
};

template <typename Op>
void testBinaryOp(Op func, BinaryOp op) {
  // c0  c1
  //  \  /
  //  node
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto t0 = full(shape, 0, dtype);
  const auto t1 = full(shape, 1, dtype);
  const auto c0 = toJitTensorBase(t0).node();
  const auto c1 = toJitTensorBase(t1).node();
  const auto tensor = func(t0, t1);
  const auto& jitTensor = toJitTensorBase(tensor);
  const auto node = &jitTensor.node()->template impl<BinaryNode>();
  ASSERT_EQ(node->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->lhs(), c0);
  ASSERT_EQ(node->rhs(), c1);
  ASSERT_EQ(node->op(), op);
  ASSERT_EQ(c0->uses(), UseValList({{node, 0}}));
  ASSERT_EQ(c1->uses(), UseValList({{node, 1}}));
}

} // namespace

TEST_F(JitTensorTest, constructor) {
  const auto data =
      ArrayFireBackend::getInstance().rand(Shape({2, 2}), dtype::f32);
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

TEST_F(JitTensorTest, full) {
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  int val = 42;
  const auto tensor = full(shape, val, dtype);
  const auto& jitTensor = toJitTensorBase(tensor);
  const auto node = jitTensor.node()->impl<ScalarNode>();
  ASSERT_EQ(node.inputs(), NodeList({}));
  ASSERT_EQ(node.uses(), UseValList({}));
  ASSERT_EQ(node.shape(), shape);
  ASSERT_EQ(node.dataType(), dtype);
  ASSERT_EQ(node.scalar<int>(), val);
}

TEST_F(JitTensorTest, add) {
  testBinaryOp(std::plus<>(), BinaryOp::Add);
}

TEST_F(JitTensorTest, sub) {
  testBinaryOp(std::minus<>(), BinaryOp::Sub);
}

TEST_F(JitTensorTest, mul) {
  testBinaryOp(std::multiplies<>(), BinaryOp::Mul);
}

TEST_F(JitTensorTest, div) {
  testBinaryOp(std::divides<>(), BinaryOp::Div);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
