/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"
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
    fl::setDefaultTensorType<JitTensor<DefaultTensorType_t>>();
  }
  TensorBackend& defaultBackend_ = DefaultTensorBackend_t::getInstance();
};

template <typename Op>
void testBinaryOp(Op func, BinaryOp op) {
  // c0  c1
  //  \  /
  //  node
  Shape shape({2, 2});
  auto dtype = dtype::s32;
  const auto t0 = full(shape, 0, dtype);
  const auto t1 = full(shape, 1, dtype);
  const auto c0 = toJitTensorBase(t0).node();
  const auto c1 = toJitTensorBase(t1).node();
  const auto tensor = func(t0, t1);
  const auto& jitTensor = toJitTensorBase(tensor);
  const auto node = jitTensor.node();
  const auto& binaryNode = node->template impl<BinaryNode>();
  ASSERT_EQ(node->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(binaryNode.lhs(), c0);
  ASSERT_EQ(binaryNode.rhs(), c1);
  ASSERT_EQ(binaryNode.op(), op);
  ASSERT_EQ(node->shape(), shape);
  ASSERT_EQ(c0->uses(), UseValList({{node, 0}}));
  ASSERT_EQ(c1->uses(), UseValList({{node, 1}}));
}

struct ShiftLeftFunctor {
  auto operator()(const Tensor& lhs, const Tensor& rhs) {
    return lhs << rhs;
  }
};

struct ShiftRightFunctor {
  auto operator()(const Tensor& lhs, const Tensor& rhs) {
    return lhs >> rhs;
  }
};

struct PowerFunctor {
  auto operator()(const Tensor& lhs, const Tensor& rhs) {
    return fl::power(lhs, rhs);
  }
};

struct MaxFunctor {
  auto operator()(const Tensor& lhs, const Tensor& rhs) {
    return fl::maximum(lhs, rhs);
  }
};

struct MinFunctor {
  auto operator()(const Tensor& lhs, const Tensor& rhs) {
    return fl::minimum(lhs, rhs);
  }
};

} // namespace

TEST_F(JitTensorTest, constructor) {
  const auto dataTensor = defaultBackend_.rand(Shape({2, 2}), dtype::f32);
  const auto data = dataTensor.toHostVector<float>();
  const Tensor tensor =
      Tensor::fromBuffer(dataTensor.shape(), data.data(), Location::Host);
  const auto& jitTensor = toJitTensorBase(tensor);
  const auto node = jitTensor.node();
  ASSERT_EQ(node->inputs(), NodeList({}));
  ASSERT_EQ(node->uses(), UseValList({}));
  ASSERT_EQ(node->shape(), dataTensor.shape());
  ASSERT_TRUE(node->isValue());
  ASSERT_TRUE(allClose(dataTensor, node->getResult().value()));
}

TEST_F(JitTensorTest, full) {
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  int val = 42;
  const auto tensor = full(shape, val, dtype);
  const auto& jitTensor = toJitTensorBase(tensor);
  const auto& node = jitTensor.node()->impl<ScalarNode>();
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

TEST_F(JitTensorTest, eq) {
  testBinaryOp(std::equal_to<>(), BinaryOp::Eq);
}

TEST_F(JitTensorTest, neq) {
  testBinaryOp(std::not_equal_to<>(), BinaryOp::Neq);
}

TEST_F(JitTensorTest, gt) {
  testBinaryOp(std::greater<>(), BinaryOp::Gt);
}

TEST_F(JitTensorTest, gte) {
  testBinaryOp(std::greater_equal<>(), BinaryOp::Gte);
}

TEST_F(JitTensorTest, lt) {
  testBinaryOp(std::less<>(), BinaryOp::Lt);
}

TEST_F(JitTensorTest, mod) {
  testBinaryOp(std::modulus<>(), BinaryOp::Mod);
}

TEST_F(JitTensorTest, and) {
  testBinaryOp(std::logical_and<>(), BinaryOp::And);
}

TEST_F(JitTensorTest, or) {
  testBinaryOp(std::logical_and<>(), BinaryOp::And);
}

TEST_F(JitTensorTest, bitwiseAnd) {
  testBinaryOp(std::bit_and<>(), BinaryOp::BitAnd);
}

TEST_F(JitTensorTest, bitwiseOr) {
  testBinaryOp(std::bit_or<>(), BinaryOp::BitOr);
}

TEST_F(JitTensorTest, bitwiseXor) {
  testBinaryOp(std::bit_xor<>(), BinaryOp::BitXor);
}

TEST_F(JitTensorTest, shiftLeft) {
  testBinaryOp(ShiftLeftFunctor(), BinaryOp::Shl);
}

TEST_F(JitTensorTest, shiftRight) {
  testBinaryOp(ShiftRightFunctor(), BinaryOp::Shr);
}

TEST_F(JitTensorTest, max) {
  testBinaryOp(MaxFunctor(), BinaryOp::Max);
}

TEST_F(JitTensorTest, min) {
  testBinaryOp(MinFunctor(), BinaryOp::Min);
}

TEST_F(JitTensorTest, pow) {
  testBinaryOp(PowerFunctor(), BinaryOp::Pow);
}

TEST_F(JitTensorTest, explicitEval) {
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto t0 = full(shape, 11, dtype);
  const auto t1 = full(shape, 22, dtype);
  auto sum = t0 + t1;
  ASSERT_FALSE(toJitTensorBase(sum).node()->getResult().has_value());
  fl::eval(sum); // explicit eval call
  ASSERT_TRUE(allClose(
      toJitTensorBase(sum).node()->getResult().value(),
      defaultBackend_.full(shape, 33, dtype)));
}

TEST_F(JitTensorTest, forcedEval) {
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  const auto t0 = full(shape, 11, dtype);
  const auto t1 = full(shape, 22, dtype);
  auto sum = t0 + t1;
  ASSERT_FALSE(toJitTensorBase(sum).node()->getResult().has_value());
  sum.toString(); // forces eval
  ASSERT_TRUE(allClose(
      toJitTensorBase(sum).node()->getResult().value(),
      defaultBackend_.full(shape, 33, dtype)));
}

TEST_F(JitTensorTest, assignment) {
  // we don't test the computation result (that's Evaluator's job) -- we only
  // test the graph we are building.

  //     Expression         The node lhs represents
  //
  // t0 = full(..., 0)      c0
  // t1 = full(..., 1)      c1
  // t2 = full(..., 2)      c2
  //
  //                        c0  c1
  //                         \  /
  // sum1 = t0 + t1          add1
  //
  //                        c0  c2
  //                         \  /
  // t0 += t2                add2
  //
  //                        c0  c2
  //                         \  /
  //                         add2  c1
  //                           \   /
  // sum3 = t0 + t1            add3
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  auto t0 = full(shape, 0, dtype);
  auto t1 = full(shape, 1, dtype);
  auto t2 = full(shape, 2, dtype);
  auto sum1 = t0 + t1;
  auto c0 = toJitTensorBase(t0).node();
  auto c1 = toJitTensorBase(t1).node();
  auto c2 = toJitTensorBase(t2).node();
  auto add1 = toJitTensorBase(sum1).node();
  t0 += t2;
  auto sum3 = t0 + t1;
  auto add2 = toJitTensorBase(t0).node();
  auto add3 = toJitTensorBase(sum3).node();

  // assignment won't affect other tensors
  ASSERT_EQ(toJitTensorBase(t1).node(), c1);
  ASSERT_EQ(toJitTensorBase(sum1).node(), add1);

  // assignment won't affect other tensor's graphs
  ASSERT_EQ(c0->inputs(), NodeList({}));
  ASSERT_EQ(c0->uses(), UseValList({{add1, 0}, {add2, 0}}));
  ASSERT_TRUE(c0->isScalar());
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{add1, 1}, {add3, 1}}));
  ASSERT_TRUE(c1->isScalar());
  ASSERT_EQ(add1->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(add1->uses(), UseValList({}));
  ASSERT_TRUE(add1->isBinary());

  // in-place add assignment creates new graph (like SSA)
  ASSERT_EQ(c2->inputs(), NodeList({}));
  ASSERT_EQ(c2->uses(), UseValList({{add2, 1}}));
  ASSERT_TRUE(c2->isScalar());
  ASSERT_EQ(add2->inputs(), NodeList({c0, c2}));
  ASSERT_EQ(add2->uses(), UseValList({{add3, 0}}));
  ASSERT_TRUE(add2->isBinary());

  // future use of assigned tensor uses its new graph
  ASSERT_EQ(add3->inputs(), NodeList({add2, c1}));
  ASSERT_EQ(add3->uses(), UseValList({}));
  ASSERT_TRUE(add3->isBinary());
}

TEST_F(JitTensorTest, assignmentWithShallowCopyAndCopy) {
  // we don't test the computation result (that's Evaluator's job) -- we only
  // test the graph we are building.

  //     Expression         The node lhs represents
  //
  // t0 = full(..., 0)        c0
  // t1 = full(..., 1)        c1
  //
  // t0c = t0.copy()          c0
  // t0sc = t0.shallowCopy()  c0
  //
  //                        c0  c1
  //                         \  /
  // t0 += t1                add
  //
  // t0c                      c0
  //
  //                        c0  c1
  //                         \  /
  // t0sc                    add
  Shape shape(Shape({2, 2}));
  auto dtype = dtype::s32;
  auto t0 = full(shape, 0, dtype);
  auto t1 = full(shape, 1, dtype);
  auto c0 = toJitTensorBase(t0).node();
  auto c1 = toJitTensorBase(t1).node();
  auto t0c = t0.copy();
  auto t0sc = toJitTensorBase(t0).shallowCopy();
  t0 += t1;
  auto add = toJitTensorBase(t0).node();

  // assignment won't affect copy
  ASSERT_EQ(toJitTensorBase(t0c).node(), c0);
  ASSERT_EQ(c0->inputs(), NodeList({}));
  ASSERT_EQ(c0->uses(), UseValList({{add, 0}}));
  ASSERT_TRUE(c0->isScalar());

  // assignment affects shallow copy
  ASSERT_EQ(toJitTensorBase(t0sc).node(), add);
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{add, 1}}));
  ASSERT_TRUE(c1->isScalar());
  ASSERT_EQ(add->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(add->uses(), UseValList({}));
  ASSERT_TRUE(add->isBinary());
}

TEST_F(JitTensorTest, indexedRead) {
  const Shape shape({5, 6, 7});
  const auto dtype = dtype::s32;
  const auto t0 = full(shape, 0, dtype);
  const std::vector<Index> indices{2, range(1, 4, 2)};
  auto indexResult = t0(indices);
  const auto indexResultCopy = indexResult.copy();
  const auto indexResultShallowCopy =
      toJitTensorBase(indexResult).shallowCopy();
  const auto c0 = toJitTensorBase(t0).node();
  const auto i0 = toJitTensorBase(indexResult).node();

  // graph is properly constructed, i.e., i0 --> c0
  ASSERT_EQ(c0->inputs(), NodeList({}));
  ASSERT_EQ(c0->uses(), UseValList({{i0, 0}}));
  ASSERT_TRUE(c0->isScalar());
  ASSERT_EQ(i0->inputs(), NodeList({c0}));
  ASSERT_EQ(i0->uses(), UseValList({}));
  ASSERT_EQ(i0->shape(), Shape({2, 7}));
  ASSERT_TRUE(i0->isIndex());
  ASSERT_EQ(i0->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i0->impl<IndexNode>().indexedNode(), c0);

  // copy/shallow-copy all yield the same node
  ASSERT_EQ(toJitTensorBase(indexResultCopy).node(), i0);
  ASSERT_EQ(toJitTensorBase(indexResultShallowCopy).node(), i0);
}

TEST_F(JitTensorTest, indexingWithOriginalDataUpdated) {
  const Shape shape(Shape({5, 6, 7}));
  const auto dtype = dtype::s32;
  auto t0 = full(shape, 0, dtype);
  const auto t1 = full(shape, 1, dtype);
  const std::vector<Index> indices{2, range(1, 4, 2)};
  const auto t0Copy = t0.copy();
  const auto t0ShallowCopy = toJitTensorBase(t0).shallowCopy();
  const auto t0Indexed = t0(indices);
  const auto t0CopyIndexed = t0Copy(indices);
  const auto t0ShallowCopyIndexed = t0ShallowCopy(indices);
  const auto c0 = toJitTensorBase(t0).node();
  t0 += t1; // affects t0, t0ShallowCopy and their indexed nodes
  const auto add = toJitTensorBase(t0).node();
  const auto c1 = toJitTensorBase(t1).node();
  const auto i0 = toJitTensorBase(t0Indexed).node();
  const auto i1 = toJitTensorBase(t0CopyIndexed).node();
  const auto i2 = toJitTensorBase(t0ShallowCopyIndexed).node();

  // shallow copy and copy have the right nodes
  ASSERT_EQ(toJitTensorBase(t0Copy).node(), c0);
  ASSERT_EQ(toJitTensorBase(t0ShallowCopy).node(), add);

  // graph is properly constructed
  //
  // c0   c1  c0
  //   \ /     |
  //   add    i1
  //    |
  //  i0/i2
  ASSERT_EQ(c0->inputs(), NodeList({}));
  ASSERT_EQ(c0->uses(), UseValList({{i1, 0}, {add, 0}}));
  ASSERT_TRUE(c0->isScalar());
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{add, 1}}));
  ASSERT_TRUE(c1->isScalar());
  ASSERT_EQ(add->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(add->uses(), UseValList({{i0, 0}, {i2, 0}}));
  ASSERT_TRUE(add->isBinary());
  ASSERT_EQ(i0->inputs(), NodeList({add}));
  ASSERT_EQ(i0->uses(), UseValList({}));
  ASSERT_EQ(i0->shape(), Shape({2, 7}));
  ASSERT_TRUE(i0->isIndex());
  ASSERT_EQ(i0->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i0->impl<IndexNode>().indexedNode(), add);
  ASSERT_EQ(i1->inputs(), NodeList({c0}));
  ASSERT_EQ(i1->uses(), UseValList({}));
  ASSERT_EQ(i1->shape(), Shape({2, 7}));
  ASSERT_TRUE(i1->isIndex());
  ASSERT_EQ(i1->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i1->impl<IndexNode>().indexedNode(), c0);
  ASSERT_EQ(i2->inputs(), NodeList({add}));
  ASSERT_EQ(i2->uses(), UseValList({}));
  ASSERT_EQ(i2->shape(), Shape({2, 7}));
  ASSERT_TRUE(i2->isIndex());
  ASSERT_EQ(i2->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i2->impl<IndexNode>().indexedNode(), add);
}

TEST_F(JitTensorTest, indexingWithIndexedDataUpdated) {
  const auto dtype = dtype::s32;
  auto t0 = full({5, 6, 7}, 0, dtype);
  const auto t1 = full({6, 7}, 1, dtype);
  const std::vector<Index> indices1{2};
  const std::vector<Index> indices2{2, range(1, 4, 2)};
  const auto t0Copy = t0.copy();
  const auto t0ShallowCopy = toJitTensorBase(t0).shallowCopy();
  const auto t0Indexed = t0(indices2);
  const auto t0CopyIndexed = t0Copy(indices2);
  const auto t0ShallowCopyIndexed = t0ShallowCopy(indices2);
  const auto c0 = toJitTensorBase(t0).node();
  t0(indices1) = t1; // affects t0, t0ShallowCopy and their indexed nodes
  const auto update = toJitTensorBase(t0).node();
  const auto c1 = toJitTensorBase(t1).node();
  const auto i0 = toJitTensorBase(t0Indexed).node();
  const auto i1 = toJitTensorBase(t0CopyIndexed).node();
  const auto i2 = toJitTensorBase(t0ShallowCopyIndexed).node();

  // shallow copy and copy have the right nodes
  ASSERT_EQ(toJitTensorBase(t0Copy).node(), c0);
  ASSERT_EQ(toJitTensorBase(t0ShallowCopy).node(), update);

  // graph is properly constructed
  //
  // indices1
  //    |
  // c0 | c1  c0
  //   \|/     |
  //  update  i1
  //    |
  //  i0/i2
  ASSERT_EQ(c0->inputs(), NodeList({}));
  ASSERT_EQ(c0->uses(), UseValList({{i1, 0}, {update, 0}}));
  ASSERT_TRUE(c0->isScalar());
  ASSERT_EQ(c1->inputs(), NodeList({}));
  ASSERT_EQ(c1->uses(), UseValList({{update, 1}}));
  ASSERT_TRUE(c1->isScalar());
  ASSERT_EQ(update->inputs(), NodeList({c0, c1}));
  ASSERT_EQ(update->uses(), UseValList({{i0, 0}, {i2, 0}}));
  ASSERT_TRUE(update->isIndexedUpdate());
  ASSERT_EQ(update->impl<IndexedUpdateNode>().indexedNode(), c0);
  ASSERT_EQ(update->impl<IndexedUpdateNode>().updateDataNode(), c1);
  ASSERT_EQ(update->impl<IndexedUpdateNode>().indexings().size(), 1);
  ASSERT_EQ(i0->inputs(), NodeList({update}));
  ASSERT_EQ(i0->uses(), UseValList({}));
  ASSERT_EQ(i0->shape(), Shape({2, 7}));
  ASSERT_TRUE(i0->isIndex());
  ASSERT_EQ(i0->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i0->impl<IndexNode>().indexedNode(), update);
  ASSERT_EQ(i1->inputs(), NodeList({c0}));
  ASSERT_EQ(i1->uses(), UseValList({}));
  ASSERT_EQ(i1->shape(), Shape({2, 7}));
  ASSERT_TRUE(i1->isIndex());
  ASSERT_EQ(i1->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i1->impl<IndexNode>().indexedNode(), c0);
  ASSERT_EQ(i2->inputs(), NodeList({update}));
  ASSERT_EQ(i2->uses(), UseValList({}));
  ASSERT_EQ(i2->shape(), Shape({2, 7}));
  ASSERT_TRUE(i2->isIndex());
  ASSERT_EQ(i2->impl<IndexNode>().indices().size(), 2);
  ASSERT_EQ(i2->impl<IndexNode>().indexedNode(), update);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  init();
  return RUN_ALL_TESTS();
}
