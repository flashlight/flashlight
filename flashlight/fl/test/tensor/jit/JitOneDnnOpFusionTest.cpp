/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/jit/Utils.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/opt/backends/onednn/OneDnnOpFusion.h"

using namespace fl;

class JitOneDnnOpFusionTest : public ::testing::Test {
 protected:
  OneDnnOpFusion oneDnnFuser_;
};

TEST_F(JitOneDnnOpFusionTest, singleScalarNode) {
  // c1
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto res = oneDnnFuser_.apply(c1);
  // nothing changes
  ASSERT_EQ(res, c1);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({}));
}

TEST_F(JitOneDnnOpFusionTest, singleBinaryNode) {
  // c1  c2
  //  \  /
  //   add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto add = BinaryNode::create(c1, c2, BinaryOp::Add);
  const auto res = oneDnnFuser_.apply(add);
  // nothing changes
  ASSERT_EQ(res, add);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{add, 0}}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{add, 1}}));
  ASSERT_EQ(add->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(add->uses(), UseValList({}));
}

TEST_F(JitOneDnnOpFusionTest, sharedBinaryNode) {
  //   c1
  //  /  \
  //  \  /
  //   sub
  //  /  \
  //  \  /
  //   add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto sub = BinaryNode::create(c1, c1, BinaryOp::Sub);
  const auto add = BinaryNode::create(sub, sub, BinaryOp::Add);
  const auto res = oneDnnFuser_.apply(add);
  // nothing changes
  ASSERT_EQ(res, add);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{sub, 0}, {sub, 1}}));
  ASSERT_EQ(sub->inputs(), NodeList({c1, c1}));
  ASSERT_EQ(sub->uses(), UseValList({{add, 0}, {add, 1}}));
  ASSERT_EQ(add->inputs(), NodeList({sub, sub}));
  ASSERT_EQ(add->uses(), UseValList({}));
}

TEST_F(JitOneDnnOpFusionTest, nonFusableRoot) {
  // c1  c2
  //  \  /
  //   mul  c3
  //    \  /
  //     add
  //      |
  //    custom
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto mul = BinaryNode::create(c1, c2, BinaryOp::Mul);
  const auto add = BinaryNode::create(mul, c3, BinaryOp::Add);
  const auto custom = CustomNode::create(
      "identity", {add}, shape, [](const std::vector<const Tensor*>& inputs) {
        return *inputs[0];
      });
  // c1  c2
  //  \  /
  //   mul  c3            c1 c2  c3       ..... (same as before)
  //    \  /               \  |  /         \ /
  //     add      ---->  fusedCustomNode   add
  //      |                   |
  //    custom              custom
  ASSERT_EQ(custom, oneDnnFuser_.apply(custom));
  const auto fusedNode = custom->inputs().at(0);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{mul, 0}, {fusedNode, 0}}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{mul, 1}, {fusedNode, 1}}));
  ASSERT_EQ(c3->inputs(), NodeList({}));
  ASSERT_EQ(c3->uses(), UseValList({{add, 1}, {fusedNode, 2}}));
  ASSERT_EQ(mul->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(mul->uses(), UseValList({{add, 0}}));
  ASSERT_EQ(add->inputs(), NodeList({mul, c3}));
  ASSERT_EQ(add->uses(), UseValList({})); // no more use, replaced by fused node
  ASSERT_EQ(fusedNode->inputs(), NodeList({c1, c2, c3}));
  ASSERT_EQ(fusedNode->uses(), UseValList({{custom, 0}}));
  ASSERT_TRUE(fusedNode->isCustom());
  ASSERT_EQ(custom->inputs(), NodeList({fusedNode}));
  ASSERT_EQ(custom->uses(), UseValList({}));
}

TEST_F(JitOneDnnOpFusionTest, nestedFusableChains) {
  // c2   c3
  //  \  /
  //   sub  c4
  //     \  /
  //  c1  div
  //   \  /
  //    mul  c5
  //     \  /
  //      add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto c1 = ScalarNode::create(shape, dtype, 1);
  const auto c2 = ScalarNode::create(shape, dtype, 2);
  const auto c3 = ScalarNode::create(shape, dtype, 3);
  const auto c4 = ScalarNode::create(shape, dtype, 4);
  const auto c5 = ScalarNode::create(shape, dtype, 5);
  const auto sub = BinaryNode::create(c2, c3, BinaryOp::Sub);
  const auto div = BinaryNode::create(sub, c4, BinaryOp::Div);
  const auto mul = BinaryNode::create(c1, div, BinaryOp::Mul);
  const auto add = BinaryNode::create(mul, c5, BinaryOp::Add);
  // c2   c3              c2  c3 c4
  //  \  /                 \  |  /
  //   sub  c4          fusedCustomNode --------    ..... (same as before)
  //     \  /                 |                |     \ /
  //  c1  div     ---->    c1 | c5         c1  |     div
  //   \  /                 \ | /           \  /
  //    mul  c5         fusedCustomRoot      mul  c5
  //     \  /                                  \  /
  //      add                                  add
  const auto fusedCustomRoot = oneDnnFuser_.apply(add);
  const auto fusedCustomNode = fusedCustomRoot->inputs().at(1);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{mul, 0}, {fusedCustomRoot, 0}}));
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{sub, 0}, {fusedCustomNode, 0}}));
  ASSERT_EQ(c3->inputs(), NodeList({}));
  ASSERT_EQ(c3->uses(), UseValList({{sub, 1}, {fusedCustomNode, 1}}));
  ASSERT_EQ(c4->inputs(), NodeList({}));
  ASSERT_EQ(c4->uses(), UseValList({{div, 1}, {fusedCustomNode, 2}}));
  ASSERT_EQ(c5->inputs(), NodeList({}));
  ASSERT_EQ(c5->uses(), UseValList({{add, 1}, {fusedCustomRoot, 2}}));
  ASSERT_EQ(sub->inputs(), NodeList({c2, c3}));
  ASSERT_EQ(sub->uses(), UseValList({{div, 0}}));
  ASSERT_EQ(div->inputs(), NodeList({sub, c4}));
  ASSERT_EQ(div->uses(), UseValList({})); // no more use, replaced by fused node
  ASSERT_EQ(mul->inputs(), NodeList({c1, fusedCustomNode}));
  ASSERT_EQ(mul->uses(), UseValList({{add, 0}}));
  ASSERT_EQ(add->inputs(), NodeList({mul, c5}));
  ASSERT_EQ(add->uses(), UseValList({}));
  ASSERT_EQ(fusedCustomNode->inputs(), NodeList({c2, c3, c4}));
  ASSERT_EQ(fusedCustomNode->uses(), UseValList({{mul, 1}, {fusedCustomRoot, 1}}));
  ASSERT_EQ(fusedCustomNode->shape(), shape);
  ASSERT_TRUE(fusedCustomNode->isCustom());
  ASSERT_EQ(fusedCustomRoot->inputs(), NodeList({c1, fusedCustomNode, c5}));
  ASSERT_EQ(fusedCustomRoot->uses(), UseValList({}));
  ASSERT_EQ(fusedCustomRoot->shape(), shape);
  ASSERT_TRUE(fusedCustomRoot->isCustom());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
