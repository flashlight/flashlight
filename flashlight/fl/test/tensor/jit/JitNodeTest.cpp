/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"
#include "flashlight/fl/tensor/backend/jit/ir/BinaryNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/CustomNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/Node.h"
#include "flashlight/fl/tensor/backend/jit/ir/ScalarNode.h"
#include "flashlight/fl/tensor/backend/jit/ir/ValueNode.h"

using namespace fl;

// things to help write tests
namespace {

struct UseVal {
  const Node* user;
  const unsigned inputIdx;
};

using NodeList = std::vector<Node*>;
using UseValList = std::vector<UseVal>;

bool operator==(UseList actual, UseValList expect) {
  if (actual.size() != expect.size()) {
    return false;
  }
  unsigned i = 0;
  for (const auto& actualUse : actual) {
    const auto& expectUse = expect[i];
    if (actualUse->user() != expectUse.user ||
        actualUse->inputIdx() != expectUse.inputIdx) {
      return false;
    }
    i++;
  }
  return true;
}

} // namespace

TEST(JitNodeTest, ScalarNodeMetaData) {
  const Shape shape({2, 2});
  const auto type = dtype::s32;
  const int value = 20;
  const auto node = ScalarNode::create(shape, type, value);
  ASSERT_EQ(node->inputs(), NodeList({}));
  ASSERT_EQ(node->getRefCount(), 0);
  ASSERT_EQ(node->uses(), UseList({}));
  ASSERT_EQ(node->isScalar(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->shape(), shape);
  ASSERT_EQ(node->dataType(), type);
  ASSERT_EQ(node->scalar<int>(), value);
  // "free" node
  delete node;
}

TEST(JitNodeTest, ValueNodeMetaData) {
  const auto tensor = full(Shape({3, 3}), 42);
  const auto node = ValueNode::create(tensor.copy());
  ASSERT_EQ(node->inputs(), NodeList({}));
  ASSERT_EQ(node->getRefCount(), 0);
  ASSERT_EQ(node->uses(), UseList({}));
  ASSERT_EQ(node->isValue(), true);
  ASSERT_TRUE(node->getResult().has_value());
  ASSERT_TRUE(allClose(node->getResult().value(), tensor));
  // "free" node
  delete node;
}

TEST(JitNodeTest, BinaryNodeMetaData) {
  const auto c1 = ScalarNode::create(Shape({2, 2}), dtype::f32, 42);
  const auto c2 = ScalarNode::create(Shape({2, 2}), dtype::f32, 42);
  const auto op = BinaryOp::Add;
  const auto node = BinaryNode::create(c1, c2, op);
  ASSERT_EQ(node->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(node->getRefCount(), 0);
  ASSERT_EQ(node->uses(), UseList({}));
  ASSERT_EQ(node->isBinary(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->lhs(), c1);
  ASSERT_EQ(node->rhs(), c2);
  ASSERT_EQ(node->op(), op);
  // "free" node
  delete node;
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
      name, {c1, c2}, [](const std::vector<const Tensor*>& inputs) -> Tensor {
        return inputs.at(0)->copy();
      });
  ASSERT_EQ(node->inputs(), NodeList({c1, c2}));
  ASSERT_EQ(node->getRefCount(), 0);
  ASSERT_EQ(node->uses(), UseList({}));
  ASSERT_EQ(node->isCustom(), true);
  ASSERT_EQ(node->getResult(), std::nullopt);
  ASSERT_EQ(node->name(), name);
  ASSERT_TRUE(allClose(node->evalFunc()({&t1, &t2}), t1));
  // "free" node
  delete node;
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

TEST(JitNodeTest, refCountUpdate) {
  Node* node = ScalarNode::create(Shape({2, 2}), dtype::s32, 20);

  ASSERT_EQ(node->getRefCount(), 0);
  node->incRefCount(); // transition to shared ownership
  ASSERT_EQ(node->getRefCount(), 1);
  node->incRefCount();
  ASSERT_EQ(node->getRefCount(), 2);
  node->decRefCount();
  ASSERT_EQ(node->getRefCount(), 1);

  // "free" nodes
  node->decRefCount();
}

TEST(JitNodeTest, refCountWithInputs) {
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
  // locally share ownership for each node
  c1->incRefCount();
  c2->incRefCount();
  add->incRefCount();
  mul->incRefCount();

  // refcount
  ASSERT_EQ(c1->getRefCount(), 2);
  ASSERT_EQ(c2->getRefCount(), 2);
  ASSERT_EQ(add->getRefCount(), 3);
  ASSERT_EQ(mul->getRefCount(), 1);

  // c1   c2
  //   \  /
  //   add
  mul->decRefCount();
  ASSERT_EQ(c1->getRefCount(), 2);
  ASSERT_EQ(c2->getRefCount(), 2);
  ASSERT_EQ(add->getRefCount(), 1);

  // c1   c2
  add->decRefCount();
  ASSERT_EQ(c1->getRefCount(), 1);
  ASSERT_EQ(c2->getRefCount(), 1);

  // "free" remaining nodes
  c1->decRefCount();
  c2->decRefCount();
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
  // locally share ownership only for root node
  mul->incRefCount();

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

  // "free" root node
  mul->decRefCount();
}

TEST(JitNodeTest, replaceAllUsesWith) {
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
  // locally share ownership only for (existing and upcoming) root nodes
  mul->incRefCount();
  c1->incRefCount();
  add1->incRefCount();
  add2->incRefCount();

  // c1   c2   c3  c4
  //   \  /     \  /
  //   add1     add2
  //            / \
  //            \ /
  //            mul
  add1->replaceAllUsesWith(add2);
  // make sure nothing got messed up
  ASSERT_EQ(c1->getRefCount(), 2);
  ASSERT_EQ(c2->getRefCount(), 1);
  ASSERT_EQ(c3->getRefCount(), 1);
  ASSERT_EQ(c4->getRefCount(), 1);
  ASSERT_EQ(add1->getRefCount(), 1);
  ASSERT_EQ(add2->getRefCount(), 3);
  ASSERT_EQ(mul->getRefCount(), 1);
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
  ASSERT_EQ(c1->getRefCount(), 1);
  ASSERT_EQ(c2->getRefCount(), 1);
  ASSERT_EQ(c3->getRefCount(), 1);
  ASSERT_EQ(c4->getRefCount(), 1);
  ASSERT_EQ(add1->getRefCount(), 1);
  ASSERT_EQ(add2->getRefCount(), 3);
  ASSERT_EQ(mul->getRefCount(), 2);
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

  // "free" locally shared node
  mul->decRefCount();
  c1->decRefCount();
  add1->decRefCount();
  add2->decRefCount();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
