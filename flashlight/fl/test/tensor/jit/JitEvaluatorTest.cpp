/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/backend/jit/JitTensor.h"
#include "flashlight/fl/tensor/backend/jit/eval/Evaluator.h"
#include "flashlight/fl/tensor/backend/jit/ir/ExternalUse.h"
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
}

TEST_F(JitEvaluatorTest, evalScalarNode) {
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  int val = 42;
  const auto tensor = full(shape, val, dtype);
  const auto node = ScalarNode::create(shape, dtype, val);
  evaluator_.eval(node);
  ASSERT_TRUE(allClose(node->getResult().value(), tensor));
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
      "addThenMul",
      {c1, c2, c3},
      shape,
      [](const std::vector<const Tensor*> inputs) {
        return (*inputs[0] + *inputs[1]) * *inputs[2];
      });
  evaluator_.eval(custom);
  ASSERT_TRUE(allClose(custom->getResult().value(), full(shape, 9, dtype)));
}

TEST_F(JitEvaluatorTest, evalIndexNodeWithoutTensorIdx) {
  const auto value = iota({4, 5, 6}, {1}, dtype::s32);
  const auto valueNode = ValueNode::create(value.copy());
  const std::vector<Index> indices{1, range(0, 3, 2), range(3)};
  const auto indexNode = IndexNode::create(valueNode, indices);
  evaluator_.eval(indexNode);
  ASSERT_TRUE(allClose(indexNode->getResult().value(), value(indices)));
  ASSERT_TRUE(allClose(valueNode->getResult().value(), value));
}

TEST_F(JitEvaluatorTest, evalIndexNodeWithTensorIdx) {
  const auto dtype = dtype::s32;
  const auto value = iota({10, 10}, {1}, dtype);
  const auto tensorIdx = iota({2, 3}, {1}, dtype);
  const auto jitTensorIdx =
      toTensor<JitTensor<DefaultTensorType_t>>(CustomNode::create(
          "createIota",
          {},
          tensorIdx.shape(),
          [tensorIdx](const std::vector<const Tensor*> /* inputs */) {
            return tensorIdx;
          }));
  const auto valueNode = ValueNode::create(value.copy());
  const auto indexNode = IndexNode::create(valueNode, {jitTensorIdx});
  evaluator_.eval(indexNode);
  const Tensor& resultTensor = indexNode->getResult().value();
  ASSERT_EQ(resultTensor.shape(), indexNode->shape());
  ASSERT_TRUE(allClose(resultTensor, value(tensorIdx)));
  ASSERT_TRUE(allClose(valueNode->getResult().value(), value));
  ASSERT_TRUE(allClose(
      toJitTensorBase(jitTensorIdx).node()->getResult().value(), tensorIdx));
}

TEST_F(JitEvaluatorTest, evalIndexedUpdateNodeWithoutTensorIdx) {
  // TODO
  // test multiple indexing update, like x(a)(b) = y (AF doesn't support this
  // atm, and OneDNN can't use `allClose` because it requires f64)
  const auto dtype = dtype::s32;
  const std::vector<Index> indices{1, range(0, 3, 2)};
  const auto indexedValue = iota({4, 5, 6}, {1}, dtype);
  const auto updateValue = iota({2, 6}, {1}, dtype);
  const auto indexedValueNode = ValueNode::create(indexedValue.copy());
  const auto updateValueNode = ValueNode::create(updateValue.copy());
  const auto indexedUpdateNode =
      IndexedUpdateNode::create(indexedValueNode, {indices}, updateValueNode);
  evaluator_.eval(indexedUpdateNode);
  // udpate expected result
  const auto resultValue = indexedValue.copy();
  resultValue(indices) = updateValue;
  ASSERT_TRUE(allClose(indexedUpdateNode->getResult().value(), resultValue));
  ASSERT_TRUE(allClose(indexedValueNode->getResult().value(), indexedValue));
  ASSERT_TRUE(allClose(updateValueNode->getResult().value(), updateValue));
}

TEST_F(JitEvaluatorTest, evalIndexedUpdateNodeWithTensorIdx) {
  const auto dtype = dtype::s32;
  const auto tensor = iota(Shape({2, 3}), {1}, dtype);
  const auto jitTensor =
      toTensor<JitTensor<DefaultTensorType_t>>(CustomNode::create(
          "createIota",
          {},
          tensor.shape(),
          [tensor](const std::vector<const Tensor*> /* inputs */) {
            return tensor;
          }));
  const auto indexedValue = iota({4, 5, 6}, {1}, dtype);
  const auto updateValue = iota({6, 5, 6}, {1}, dtype);
  const auto indexedValueNode = ValueNode::create(indexedValue.copy());
  const auto updateValueNode = ValueNode::create(updateValue.copy());
  const auto indexedUpdateNode = IndexedUpdateNode::create(
      indexedValueNode, {{jitTensor}}, updateValueNode);
  evaluator_.eval(indexedUpdateNode);
  // udpate expected result
  const auto resultValue = indexedValue.copy();
  resultValue({tensor}) = updateValue;
  ASSERT_TRUE(allClose(indexedUpdateNode->getResult().value(), resultValue));
  ASSERT_TRUE(allClose(indexedValueNode->getResult().value(), indexedValue));
  ASSERT_TRUE(allClose(updateValueNode->getResult().value(), updateValue));
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
  ExternalUse u1(c1); // this forces evaluator to keep c1's result
  evaluator_.eval(add);
  ASSERT_TRUE(allClose(c1->getResult().value(), full(shape, 1, dtype)));
  ASSERT_FALSE(c2->getResult().has_value()); // result not retained
  ASSERT_TRUE(allClose(add->getResult().value(), full(shape, 3, dtype)));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
