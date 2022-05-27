/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace ::testing;
using namespace fl;

TEST(TensorBinaryOpsTest, BinaryOperators) {
  // TODO:{fl::Tensor}{testing} expand this test/add a million fixtures, etc
  auto a = fl::full({2, 2}, 1);
  auto b = fl::full({2, 2}, 2);
  auto c = fl::full({2, 2}, 3);

  ASSERT_TRUE(allClose((a == b), (b == c)));
  ASSERT_TRUE(allClose((a + b), c));
  ASSERT_TRUE(allClose((c - b), a));
  ASSERT_TRUE(allClose((c * b), fl::full({2, 2}, 6)));

  auto d = fl::full({4, 5, 6}, 6.);
  ASSERT_THROW(a + d, std::invalid_argument);
  ASSERT_THROW(a + fl::full({7, 8}, 9.), std::exception);
}

TEST(TensorBinaryOpsTest, minimum) {
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 3}, 2);
  auto c = fl::minimum(a, b);
  ASSERT_EQ(a.type(), c.type());
  ASSERT_TRUE(allClose(a, c));
  ASSERT_TRUE(allClose(fl::minimum(1, b).astype(a.type()), a));
  ASSERT_TRUE(allClose(fl::minimum(b, 1).astype(a.type()), a));
}

TEST(TensorBinaryOpsTest, maximum) {
  auto a = fl::full({3, 3}, 1);
  auto b = fl::full({3, 3}, 2);
  auto c = fl::maximum(a, b);
  ASSERT_EQ(b.type(), c.type());
  ASSERT_TRUE(allClose(b, c));
  ASSERT_TRUE(allClose(fl::maximum(1, b).astype(a.type()), b));
  ASSERT_TRUE(allClose(fl::maximum(b, 1).astype(a.type()), b));
}

using binaryOpFunc_t = Tensor (*)(const Tensor& lhs, const Tensor& rhs);

TEST(TensorBinaryOpsTest, broadcasting) {
  // Collection of {lhs, rhs, tileShapeLhs, tileShapeRhs} corresponding to
  // broadcasting [lhs] to [rhs] by tiling by the the respective tileShapes
  struct ShapeData {
    Shape lhs; // broadcast from
    Shape rhs; // broadcast to
    Shape tileShapeLhs;
    Shape tileShapeRhs;
  };
  std::vector<ShapeData> shapes = {
      {{3, 1}, {3, 3}, {1, 3}, {1, 1}},
      {{3}, {3, 3}, {1, 3}, {1, 1}},
      {{3, 1, 4}, {3, 6, 4}, {1, 6, 1}, {1, 1, 1}},
      {{3, 1, 4, 1}, {3, 2, 4, 5}, {1, 2, 1, 5}, {1, 1, 1, 1}},
      {{1, 10}, {8, 10}, {8, 1}, {1, 1}},
      {{2, 1, 5, 1}, {2, 3, 5, 3}, {1, 3, 1, 3}, {1, 1, 1, 1}},
      {{3, 1, 2, 1}, {1, 4, 1, 5}, {1, 4, 1, 5}, {3, 1, 2, 1}},
      {{3, 2, 1}, {3, 1, 4, 1}, {1, 1, 4}, {1, 2, 1, 1}}};

  std::unordered_map<binaryOpFunc_t, std::string> functions = {
      {fl::minimum, "minimum"},
      {fl::maximum, "maximum"},
      {fl::power, "power"},
      {fl::add, "add"},
      {fl::add, "add"},
      {fl::sub, "sub"},
      {fl::mul, "mul"},
      {fl::div, "div"},
      {fl::eq, "eq"},
      {fl::neq, "neq"},
      {fl::lessThan, "lessThan"},
      {fl::lessThanEqual, "lessThanEqual"},
      {fl::greaterThan, "greaterThan"},
      {fl::greaterThanEqual, "greaterThanEqual"},
      {fl::logicalOr, "logicalOr"},
      {fl::logicalAnd, "logicalAnd"},
      {fl::mod, "mod"},
      {fl::bitwiseOr, "bitwiseOr"},
      {fl::bitwiseXor, "bitwiseXor"},
      {fl::lShift, "lShift"},
      {fl::rShift, "rShift"}};

  auto doBinaryOp = [](const Tensor& lhs,
                       const Tensor& rhs,
                       const Shape& tileShapeLhs,
                       const Shape& tileShapeRhs,
                       binaryOpFunc_t func) -> std::pair<Tensor, Tensor> {
    assert(lhs.ndim() <= rhs.ndim());
    return {
        func(lhs, rhs), func(tile(lhs, tileShapeLhs), tile(rhs, tileShapeRhs))};
  };

  auto computeBroadcastShape = [](const Shape& lhsShape,
                                  const Shape& rhsShape) -> Shape {
    unsigned maxnDim = std::max(lhsShape.ndim(), rhsShape.ndim());
    Shape outShape{std::vector<Dim>(maxnDim)};
    for (unsigned i = 0; i < maxnDim; ++i) {
      if (i > lhsShape.ndim() - 1) {
        outShape[i] = rhsShape[i];
      } else if (i > rhsShape.ndim() - 1) {
        outShape[i] = lhsShape[i];
      } else if (lhsShape[i] == 1) {
        outShape[i] = rhsShape[i];
      } else if (rhsShape[i] == 1) {
        outShape[i] = lhsShape[i];
      } else if (lhsShape[i] == rhsShape[i]) {
        outShape[i] = lhsShape[i];
      } else if (lhsShape[i] != rhsShape[i]) {
        throw std::runtime_error(
            "computeBroadcastShape - cannot broadcast shape");
      }
    }
    return outShape;
  };

  for (auto funcp : functions) {
    for (auto& shapeData : shapes) {
      auto lhs = ((fl::rand(shapeData.lhs) + 1) * 10).astype(fl::dtype::s32);
      auto rhs = ((fl::rand(shapeData.rhs) + 1) * 10).astype(fl::dtype::s32);

      auto [actualOut, expectedOut] = doBinaryOp(
          lhs,
          rhs,
          shapeData.tileShapeLhs,
          shapeData.tileShapeRhs,
          funcp.first);

      Shape expectedShape = computeBroadcastShape(shapeData.lhs, shapeData.rhs);

      std::stringstream ss;
      ss << "lhs: " << shapeData.lhs << " rhs: " << shapeData.rhs
         << " function: " << funcp.second;
      auto testData = ss.str();

      ASSERT_EQ(actualOut.shape(), expectedShape) << testData;
      ASSERT_TRUE(allClose(actualOut, expectedOut)) << testData;
    }

    // Scalar broadcasting
    const double scalarVal = 4;
    const Shape inShape = {2, 3, 4};
    const auto lhs = fl::rand(inShape).astype(fl::dtype::s32);
    const auto rhs = fl::fromScalar(scalarVal, fl::dtype::s32);
    const auto rhsTiled = fl::full(inShape, scalarVal, fl::dtype::s32);
    ASSERT_TRUE(allClose(funcp.first(lhs, rhs), funcp.first(lhs, rhsTiled)));
  }
}

TEST(TensorBinaryOpsTest, power) {
  auto a = fl::full({3, 3}, 2.);
  auto b = fl::full({3, 3}, 2.);
  ASSERT_TRUE(allClose(fl::power(a, b), a * b));
}

TEST(TensorBinaryOpsTest, powerDouble) {
  auto a = fl::full({3, 3}, 2.);
  ASSERT_TRUE(allClose(fl::power(a, 3), a * a * a));

  auto b = fl::full({3, 3}, 2.);
  ASSERT_TRUE(
      allClose(fl::power(3, a), fl::full(b.shape(), 3 * 3, fl::dtype::f32)));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
