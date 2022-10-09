/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

using namespace fl;

class JitEvaluatorTest : public ::testing::Test {
 protected:
  JitEvaluatorTest() : evaluator_(DefaultTensorBackend_t::getInstance()) {}
  Evaluator evaluator_;
};

TEST_F(JitEvaluatorTest, evalValueNode) {
  const auto tensor = fl::rand(Shape({2, 2}), dtype::f32);
  const auto node = ValueNode::create(tensor.copy());
  evaluator_.eval(node);
  ASSERT_TRUE(allClose(node->getResult().value(), tensor));
  // node is owned locally (didn't transition to shared ownership)
  delete node;
}

TEST_F(JitEvaluatorTest, evalScalarNode) {
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  int val = 42;
  const auto tensor = full(shape, val, dtype);
  const auto node = ScalarNode::create(shape, dtype, val);
  evaluator_.eval(node);
  ASSERT_TRUE(allClose(node->getResult().value(), tensor));
  // node is owned locally (didn't transition to shared ownership)
  delete node;
}

TEST_F(JitEvaluatorTest, evalBinaryNode) {
  // c1  c2
  //  \  /
  //   add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto add = BinaryNode::create(c1, c2, BinaryOp::Add);
  evaluator_.eval(add);
  ASSERT_TRUE(allClose(add->getResult().value(), full(shape, 3, dtype)));
  // root node is owned locally (didn't transition to shared ownership)
  delete add;
}

TEST_F(JitEvaluatorTest, evalCustomNode) {
  // c1  c2  c3
  //  \  |  /
  //   custom
  //
  // evaluation logic: (c1 + c2) * c3
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto custom = CustomNode::create(
      "addThenMul", {c1, c2, c3}, [](const std::vector<const Tensor*> inputs) {
        return (*inputs[0] + *inputs[1]) * *inputs[2];
      });
  evaluator_.eval(custom);
  ASSERT_TRUE(allClose(custom->getResult().value(), full(shape, 9, dtype)));
  // root node is owned locally (didn't transition to shared ownership)
  delete custom;
}

TEST_F(JitEvaluatorTest, evalSharedInput) {
  //   c1
  //  /  \
  //  \  /
  //   add
  // Sanity check
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto add = BinaryNode::create(c1, c1, BinaryOp::Add);
  evaluator_.eval(add);
  ASSERT_TRUE(allClose(add->getResult().value(), full(shape, 2, dtype)));
  // root node is owned locally (didn't transition to shared ownership)
  delete add;
}

TEST_F(JitEvaluatorTest, evalRetainResults) {
  // c1  c2
  //  \  /
  //   add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto add = BinaryNode::create(c1, c2, BinaryOp::Add);
  c1->incRefCount(); // this forces evaluator to retain result
  evaluator_.eval(add);
  ASSERT_TRUE(allClose(c1->getResult().value(), full(shape, 1, dtype)));
  ASSERT_FALSE(c2->getResult().has_value());
  ASSERT_TRUE(allClose(add->getResult().value(), full(shape, 3, dtype)));
  // root node is owned locally (didn't transition to shared ownership)
  delete add;
  c1->decRefCount();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
