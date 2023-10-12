/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/jit/JitTensor.h"
#include "flashlight/fl/tensor/backend/jit/Utils.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/IndexedUpdateNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

using namespace fl;

TEST(JitNodeTest, ScalarNodeMetaData) {
  const Shape shape({2, 2});
  const auto type = dtype::s32;
  const int value = 20;
  const auto node = ScalarNode::create(shape, type, value);
  ASSERT_EQ(node->inputs(), NodeList({}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isScalar(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->shape(), shape);
  ASSERT_EQ(node->dataType(), type);
  ASSERT_EQ(node->scalar<int>(), value);
}

TEST(JitNodeTest, ValueNodeMetaData) {
  const auto tensor = full(Shape({3, 3}), 42);
  const auto node = ValueNode::create(tensor.copy());
  ASSERT_EQ(node->inputs(), NodeList({}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isValue(), true);
  ASSERT_EQ(node->shape(), tensor.shape());
  ASSERT_TRUE(node->getResult().has_value());
  ASSERT_TRUE(allClose(node->getResult().value(), tensor));
}

TEST(JitNodeTest, BinaryNodeMetaData) {
  Shape lhsShape({1, 4});
  Shape rhsShape({3, 1, 5});
  Shape outShape({3, 4, 5}); // test broadcast
  const auto c1 = ScalarNode::create(lhsShape, dtype::f32, 42);
  const auto c2 = ScalarNode::create(rhsShape, dtype::f32, 42);
  const auto op = BinaryOp::Add;
  const auto node = BinaryNode::create(c1, c2, op);
  ASSERT_EQ(node->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isBinary(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->lhs(), c1);
  ASSERT_EQ(node->rhs(), c2);
  ASSERT_EQ(node->op(), op);
  ASSERT_EQ(node->shape(), outShape);
}

TEST(JitNodeTest, CustomNodeMetaData) {
  Shape shape({2, 2});
  auto type = dtype::f32;
  const auto c1 = ScalarNode::create(shape, type, 42);
  const auto c2 = ScalarNode::create(shape, type, 23);
  const auto t1 = full(shape, 42, type);
  const auto t2 = full(shape, 23, type);
  const auto name = "foobar";
  const auto node = CustomNode::create(
      name,
      {c1, c2},
      shape,
      [](const std::vector<const Tensor*>& inputs) -> Tensor {
        return inputs.at(0)->copy();
      });
  ASSERT_EQ(node->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isCustom(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->name(), name);
  ASSERT_EQ(node->shape(), shape);
  ASSERT_TRUE(allClose(node->evalFunc()({&t1, &t2}), t1));
}

TEST(JitNodeTest, IndexNodeMetaData) {
  const auto c0 = ScalarNode::create({5, 6, 7}, dtype::f32, 0);
  std::vector<Index> indices{1, range(0, 3, 2)};
  const auto node = IndexNode::create(c0, indices);
  ASSERT_EQ(node->inputs(), NodeList({c0}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isIndex(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->indexedNode(), c0);
  ASSERT_EQ(node->indices().size(), 2); // can't check equality easily...
  // - 1 reduces first dimension -- {6, 7}
  // - [0:3:2] takes 2 elements  -- {2, 7}
  ASSERT_EQ(node->shape(), Shape({2, 7}));
}

TEST(JitNodeTest, IndexNodeSameShapeTensorIndices) {
  const Shape shape({5, 6, 7});
  const auto c0 = ScalarNode::create(shape, dtype::f32, 0);
  const auto t1 =
      JitTensor<DefaultTensorType_t>().backend().full(shape, 1, dtype::f32);
  const auto c1 = toJitTensorBase(t1).node();
  std::vector<Index> indices{t1};
  const auto node = IndexNode::create(c0, indices);
  ASSERT_EQ(node->inputs(), NodeList({c0, c1})); // includes tensor index
  ASSERT_EQ(node->indexedNode(), c0);
  ASSERT_EQ(node->indices().size(), 1); // can't check equality easily...
  // if tensor index has same shape as indexed tensor, output shape is simply
  // the flattened tensor shape
  ASSERT_EQ(node->shape(), Shape({210}));
}

TEST(JitNodeTest, IndexNodeDifferentShapeTensorIndices) {
  const auto c0 = ScalarNode::create({5, 6, 7}, dtype::f32, 0);
  const auto t1 =
      JitTensor<DefaultTensorType_t>().backend().full({2, 3}, 1, dtype::f32);
  const auto c1 = toJitTensorBase(t1).node();
  std::vector<Index> indices{t1};
  const auto node = IndexNode::create(c0, indices);
  ASSERT_EQ(node->inputs(), NodeList({c0, c1})); // includes tensor index
  ASSERT_EQ(node->indexedNode(), c0);
  ASSERT_EQ(node->indices().size(), 1); // can't check equality easily...
  // tensor index shape flattened as base, then reduce 1 dimension
  ASSERT_EQ(node->shape(), Shape({6, 6, 7}));
}

TEST(JitNodeTest, IndexedUpdateNodeMetaData) {
  const auto c0 = ScalarNode::create(Shape({5, 6, 7}), dtype::f32, 0);
  const auto c1 = ScalarNode::create(Shape({2, 3, 6, 7}), dtype::f32, 0);
  const auto t2 = JitTensor<DefaultTensorType_t>().backend().full(
      Shape({2, 3}), 1, dtype::f32);
  const auto c2 = toJitTensorBase(t2).node();
  std::vector<Index> indices{t2};
  const auto node = IndexedUpdateNode::create(c0, {indices}, c1);
  ASSERT_EQ(node->inputs(), NodeList({c0, c1, c2})); // includes tensor index
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->isIndexedUpdate(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->indexedNode(), c0);
  ASSERT_EQ(node->indexings().size(), 1); // can't check equality easily...
  ASSERT_EQ(node->shape(), c0->shape());
}

TEST(JitNodeTest, getSetResult) {
  const auto node = ScalarNode::create(Shape({2, 2}), dtype::f32, 42);
  const auto tensor = full(Shape({2, 2}), 23, dtype::s32);
  // NOTE it's client's responsibility to ensure the result is correct
  node->setResult(tensor.copy());
  ASSERT_TRUE(node->getResult().has_value());
  ASSERT_TRUE(allClose(node->getResult().value(), tensor));
  ASSERT_THROW(node->setResult(tensor.copy()), std::invalid_argument);
}

TEST(JitNodeTest, inputsAndUses) {
  // c1   c2
  //   \  /
  //   add
  //   / \
  //   \ /
  //   mul
  Shape shape({2, 2, 2});
  auto type = dtype::s32;
  auto c1 = ScalarNode::create(shape, type, 1);
  auto c2 = ScalarNode::create(shape, type, 2);
  auto add = BinaryNode::create(c1, c2, BinaryOp::Add);
  auto mul = BinaryNode::create(add, add, BinaryOp::Mul);

  // inputs
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(add->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(mul->inputs(), NodeList({add, add}));

  // uses
  ASSERT_EQ(c1->uses(), UseValList({{add, 0}}));
  ASSERT_EQ(c2->uses(), UseValList({{add, 1}}));
  ASSERT_EQ(add->uses(), UseValList({{mul, 0}, {mul, 1}}));
  ASSERT_EQ(mul->uses(), UseValList({}));
}

TEST(JitNodeTest, replaceAllUsesWithRootNodes) {
  // c1  c2
  Shape shape({2, 2, 2});
  auto type = dtype::s32;
  auto c1 = ScalarNode::create(shape, type, 1);
  auto c2 = ScalarNode::create(shape, type, 2);

  // c1  c2
  // --->
  // c1  c2
  c1->replaceAllUsesWith(c2);
  // nothing changed
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({}));
  ASSERT_EQ(c2->uses(), UseValList({}));
}

TEST(JitNodeTest, replaceAllUsesWithSharedNodes) {
  // c1   c2   c3  c4
  //   \  /     \  /
  //   add1     add2
  //   / \
  //   \ /
  //   mul
  Shape shape({2, 2, 2});
  auto type = dtype::s32;
  auto c1 = ScalarNode::create(shape, type, 1);
  auto c2 = ScalarNode::create(shape, type, 2);
  auto c3 = ScalarNode::create(shape, type, 3);
  auto c4 = ScalarNode::create(shape, type, 4);
  auto add1 = BinaryNode::create(c1, c2, BinaryOp::Add);
  auto add2 = BinaryNode::create(c3, c4, BinaryOp::Add);
  auto mul = BinaryNode::create(add1, add1, BinaryOp::Mul);

  // c1   c2   c3  c4
  //   \  /     \  /
  //   add1     add2
  //            / \
  //            \ /
  //            mul
  add1->replaceAllUsesWith(add2);
  // make sure nothing got messed up
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c3->inputs(), NodeList({}));
  ASSERT_EQ(c4->inputs(), NodeList({}));
  ASSERT_EQ(add1->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(add2->inputs(), NodeList({c3, c4}));
  ASSERT_EQ(mul->inputs(), NodeList({add2, add2}));
  ASSERT_EQ(c1->uses(), UseValList({{add1, 0}}));
  ASSERT_EQ(c2->uses(), UseValList({{add1, 1}}));
  ASSERT_EQ(c3->uses(), UseValList({{add2, 0}}));
  ASSERT_EQ(c4->uses(), UseValList({{add2, 1}}));
  ASSERT_EQ(add1->uses(), UseValList({}));
  ASSERT_EQ(add2->uses(), UseValList({{mul, 0}, {mul, 1}}));
  ASSERT_EQ(mul->uses(), UseValList({}));

  // c1   c3  c4
  //       \  /
  //       add2
  //       / \
  //       \ /
  //       mul  c2
  //        \   /
  //        add1
  c1->replaceAllUsesWith(mul);
  // make sure nothing got messed up
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c3->inputs(), NodeList({}));
  ASSERT_EQ(c4->inputs(), NodeList({}));
  ASSERT_EQ(add1->inputs(), NodeList({mul, c2}));
  ASSERT_EQ(add2->inputs(), NodeList({c3, c4}));
  ASSERT_EQ(mul->inputs(), NodeList({add2, add2}));
  ASSERT_EQ(c1->uses(), UseValList({}));
  ASSERT_EQ(c2->uses(), UseValList({{add1, 1}}));
  ASSERT_EQ(c3->uses(), UseValList({{add2, 0}}));
  ASSERT_EQ(c4->uses(), UseValList({{add2, 1}}));
  ASSERT_EQ(add1->uses(), UseValList({}));
  ASSERT_EQ(add2->uses(), UseValList({{mul, 0}, {mul, 1}}));
  ASSERT_EQ(mul->uses(), UseValList({{add1, 0}}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
