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
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/jit/Utils.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"
#include "flashlight/fl/tensor/backend/jit/opt/passes/ScalarFolding.h"

using namespace fl;

class JitScalarFoldingTest : public ::testing::Test {
 protected:
  TensorBackend& defaultBackend_ = DefaultTensorBackend_t::getInstance();
  ScalarFolding scalarFolder_;
};

TEST_F(JitScalarFoldingTest, identity) {
  // v1  c2
  //  \  /
  //   add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto v1 = ValueNode::create(defaultBackend_.full(shape, 1, dtype));
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto add = BinaryNode::create(v1, c2, BinaryOp::Add);
  // nothing changed
  ASSERT_EQ(add, scalarFolder_.apply(add));
  ASSERT_EQ(add->inputs(), NodeList({v1, c2}));
  ASSERT_EQ(add->uses(), UseValList({}));
  ASSERT_EQ(v1->inputs(), NodeList({}));
  ASSERT_EQ(v1->uses(), UseValList({{add, 0}}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{add, 1}}));
}

TEST_F(JitScalarFoldingTest, binaryNode) {
  // c1  c2
  //  \  /
  //   add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto add = BinaryNode::create(c1, c2, BinaryOp::Add);
  const auto res = scalarFolder_.apply(add);
  // c1  c2        c1  c2
  //  \  /   --->   \  /
  //   add           add  res
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{add, 0}}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{add, 1}}));
  ASSERT_EQ(add->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(add->uses(), UseValList({}));
  ASSERT_EQ(res->inputs(), NodeList({}));
  ASSERT_EQ(res->uses(), UseValList({}));
  ASSERT_EQ(res->impl<ScalarNode>().shape(), shape);
  ASSERT_EQ(res->impl<ScalarNode>().dataType(), dtype);
  ASSERT_EQ(res->impl<ScalarNode>().scalar<int>(), 3);
}

TEST_F(JitScalarFoldingTest, sharedInput) {
  //   c1
  //  /  \
  //  \  /
  //   sub
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto add = BinaryNode::create(c1, c1, BinaryOp::Add);
  const auto res = scalarFolder_.apply(add);
  //   c1            c1
  //  /  \          /  \
  //  \  /   --->   \  /
  //   add           add  res
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{add, 0}, {add, 1}}));
  ASSERT_EQ(add->inputs(), NodeList({c1, c1}));
  ASSERT_EQ(add->uses(), UseValList({}));
  ASSERT_EQ(res->inputs(), NodeList({}));
  ASSERT_EQ(res->uses(), UseValList({}));
  ASSERT_EQ(res->impl<ScalarNode>().shape(), shape);
  ASSERT_EQ(res->impl<ScalarNode>().dataType(), dtype);
  ASSERT_EQ(res->impl<ScalarNode>().scalar<int>(), 2);
}

TEST_F(JitScalarFoldingTest, multipleUsers) {
  //    c2 c3
  //     \ /
  // c1  add  c4
  //   \ / \  /
  //   sub   mul
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto c4 = ScalarNode::create(shape, dtype, 4);
  const auto add = BinaryNode::create(c2, c3, BinaryOp::Add);
  const auto sub = BinaryNode::create(c1, add, BinaryOp::Sub);
  const auto mul = BinaryNode::create(add, c4, BinaryOp::Mul);
  const auto res = scalarFolder_.apply(mul);
  //    c2 c3
  //     \ /
  // c1  add  c4 ---> c1   c5  c4
  //   \ / \  /         \ / \  /
  //   sub   mul        sub   mul    res
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{sub, 0}}));
  ASSERT_EQ(c4->inputs(), NodeList({}));
  ASSERT_EQ(c4->uses(), UseValList({{mul, 1}}));
  // intermediate node (add --> c5) is optimized too, and benefits all users
  const auto c5 = sub->inputs().at(1);
  ASSERT_EQ(c5->inputs(), NodeList({}));
  ASSERT_EQ(c5->uses(), UseValList({{sub, 1}, {mul, 0}}));
  ASSERT_EQ(c5->impl<ScalarNode>().shape(), shape);
  ASSERT_EQ(c5->impl<ScalarNode>().dataType(), dtype);
  ASSERT_EQ(c5->impl<ScalarNode>().scalar<int>(), 5);
  ASSERT_EQ(sub->inputs(), NodeList({c1, c5}));
  ASSERT_EQ(sub->uses(), UseValList({}));
  ASSERT_EQ(mul->inputs(), NodeList({c5, c4}));
  ASSERT_EQ(mul->uses(), UseValList({}));
  ASSERT_EQ(res->inputs(), NodeList({}));
  ASSERT_EQ(res->uses(), UseValList({}));
  ASSERT_EQ(res->impl<ScalarNode>().shape(), shape);
  ASSERT_EQ(res->impl<ScalarNode>().dataType(), dtype);
  ASSERT_EQ(res->impl<ScalarNode>().scalar<int>(), 20);
}

TEST_F(JitScalarFoldingTest, nonFoldableRoot) {
  // c6  c3
  //  \  /
  //   div
  //    |
  //  custom
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c6 = ScalarNode::create(shape, dtype, 6);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto div = BinaryNode::create(c6, c3, BinaryOp::Div);
  const auto custom = CustomNode::create(
      "identity", {div}, shape, [](const std::vector<const Tensor*>& inputs) {
      return *inputs[0];
      });
  // c6  c3
  //  \  /
  //   div   --->    c2
  //    |             |
  //  custom       custom
  ASSERT_EQ(custom, scalarFolder_.apply(custom));
  // intermediate node (add --> c5) is optimized too
  const auto c2 = custom->inputs().at(0);
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{custom, 0}}));
  ASSERT_EQ(c2->impl<ScalarNode>().shape(), shape);
  ASSERT_EQ(c2->impl<ScalarNode>().dataType(), dtype);
  ASSERT_EQ(c2->impl<ScalarNode>().scalar<int>(), 2);
  ASSERT_EQ(c2->impl<ScalarNode>().scalar<int>(), 2);
  ASSERT_EQ(custom->inputs(), NodeList({c2}));
  ASSERT_EQ(custom->uses(), UseValList({}));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
