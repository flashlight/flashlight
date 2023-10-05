/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <functional>
#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace ::testing;
using namespace fl;

namespace {
// Always cast towards potentially signed type because otherwise ArrayFire may
// clip values, e.g., negative casting to unsigned becomse 0.
template <typename ScalarType, typename Op>
void assertTensorScalarBinop(
    const Tensor& in,
    ScalarType scalar,
    Op op,
    const Tensor& expectOut) {
  auto result = op(in, scalar);
  auto expect = expectOut.astype(result.type());
  ASSERT_TRUE(allClose(result, expect))
      << "in.type(): " << in.type()
      << ", ScalarType: " << dtype_traits<ScalarType>::getName();
}

template <typename ScalarType, typename Op>
void assertScalarTensorBinop(
    ScalarType scalar,
    const Tensor& in,
    Op op,
    const Tensor& expectOut) {
  auto result = op(scalar, in);
  auto expect = expectOut.astype(result.type());
  ASSERT_TRUE(allClose(result, expect))
      << "ScalarType: " << dtype_traits<ScalarType>::getName()
      << ", in.type(): " << in.type();
}

template <typename Op>
void assertScalarTensorCommutativeBinop(
    char scalar,
    const Tensor& in,
    Op op,
    const Tensor& out) {
  assertScalarTensorBinop(scalar, in, op, out);
  assertTensorScalarBinop(in, scalar, op, out);
}

template <typename Op>
void assertCommutativeBinop(
    const Tensor& in1,
    const Tensor& in2,
    Op op,
    const Tensor& out) {
  ASSERT_TRUE(allClose(op(in1, in2), out))
      << "in1.type(): " << in1.type() << ", in2.type(): " << in2.type();
  ASSERT_TRUE(allClose(op(in2, in1), out))
      << "in1.type(): " << in1.type() << ", in2.type(): " << in2.type();
}

void applyToAllFpDtypes(std::function<void(fl::dtype)> func) {
  func(dtype::f16);
  func(dtype::f32);
  func(dtype::f64);
}

void applyToAllIntegralDtypes(std::function<void(fl::dtype)> func) {
  // TODO casting to `b8` clips values to 0 and 1, which breaks the fixtures
  // func(dtype::b8);
  func(dtype::u8);
  func(dtype::s16);
  func(dtype::u16);
  func(dtype::s32);
  func(dtype::u32);
  func(dtype::s64);
  func(dtype::u64);
}

void applyToAllDtypes(std::function<void(fl::dtype)> func) {
  applyToAllFpDtypes(func);
  applyToAllIntegralDtypes(func);
}
} // namespace

TEST(TensorBinaryOpsTest, ArithmeticBinaryOperators) {
  auto testArithmeticBinops = [](dtype type) {
    auto a = Tensor::fromVector<float>({2, 2}, {0, 1, 2, 3}).astype(type);
    auto b = Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4}).astype(type);
    auto c = Tensor::fromVector<float>({2, 2}, {1, 3, 5, 7}).astype(type);
    auto d = Tensor::fromVector<float>({2, 2}, {1, 6, 15, 28}).astype(type);
    auto e = Tensor::fromVector<float>({2, 2}, {3, 2, 1, 0}).astype(type);
    auto f = Tensor::fromVector<float>({2, 2}, {2, 4, 6, 8}).astype(type);
    auto z = fl::full({2, 2}, 0, type);

    assertCommutativeBinop(a, z, std::plus<>(), a);
    assertCommutativeBinop(a, b, std::plus<>(), c);
    assertScalarTensorCommutativeBinop(1, a, std::plus<>(), b);
    assertScalarTensorCommutativeBinop(0, a, std::plus<>(), a);

    ASSERT_TRUE(allClose((c - z), c)) << "dtype: " << type;
    ASSERT_TRUE(allClose((z - c), -c)) << "dtype: " << type;
    ASSERT_TRUE(allClose((c - b), a)) << "dtype: " << type;
    assertTensorScalarBinop(b, 1, std::minus<>(), a);
    assertScalarTensorBinop(3, a, std::minus<>(), e);
    assertTensorScalarBinop(a, 0, std::minus<>(), a);

    assertCommutativeBinop(c, z, std::multiplies<>(), z);
    assertCommutativeBinop(c, b, std::multiplies<>(), d);
    assertScalarTensorCommutativeBinop(0, a, std::multiplies<>(), z);
    assertScalarTensorCommutativeBinop(1, a, std::multiplies<>(), a);
    assertScalarTensorCommutativeBinop(2, b, std::multiplies<>(), f);

    ASSERT_TRUE(allClose((z / b), z)) << "dtype: " << type;
    ASSERT_TRUE(allClose((d / b), c)) << "dtype: " << type;
    assertTensorScalarBinop(z, 1, std::divides<>(), z);
    assertTensorScalarBinop(a, 1, std::divides<>(), a);
    assertTensorScalarBinop(f, 2, std::divides<>(), b);
    // TODO division by zero doesn't always fail.
    // e.g., ArrayFire yields max value of dtype
  };

  applyToAllDtypes(testArithmeticBinops);
}

TEST(TensorBinaryOpsTest, ComparisonBinaryOperators) {
  auto falses = fl::full({2, 2}, 0, dtype::b8);
  auto trues = fl::full({2, 2}, 1, dtype::b8);
  auto falseTrues =
      Tensor::fromVector<float>({2, 2}, {0, 1, 0, 1}).astype(fl::dtype::b8);
  auto trueFalses =
      Tensor::fromVector<float>({2, 2}, {1, 0, 1, 0}).astype(fl::dtype::b8);

  auto testComparisonBinops = [&](dtype type) {
    auto a = Tensor::fromVector<float>({2, 2}, {0, 1, 2, 3}).astype(type);
    auto b = Tensor::fromVector<float>({2, 2}, {0, 0, 2, 0}).astype(type);
    auto c = Tensor::fromVector<float>({2, 2}, {2, 3, 4, 5}).astype(type);
    auto d = Tensor::fromVector<float>({2, 2}, {0, 4, 2, 6}).astype(type);
    auto e = Tensor::fromVector<float>({2, 2}, {0, 1, 0, 1}).astype(type);

    ASSERT_TRUE(allClose((a == a), trues)) << "dtype: " << type;
    assertCommutativeBinop(a, b, std::equal_to<>(), trueFalses);
    assertCommutativeBinop(a, c, std::equal_to<>(), falses);
    assertScalarTensorCommutativeBinop(4, a, std::equal_to<>(), falses);
    assertScalarTensorCommutativeBinop(1, e, std::equal_to<>(), falseTrues);

    ASSERT_TRUE(allClose((a != a), falses)) << "dtype: " << type;
    assertCommutativeBinop(a, b, std::not_equal_to<>(), falseTrues);
    assertCommutativeBinop(a, c, std::not_equal_to<>(), trues);
    assertScalarTensorCommutativeBinop(4, a, std::not_equal_to<>(), trues);
    assertScalarTensorCommutativeBinop(1, e, std::not_equal_to<>(), trueFalses);

    ASSERT_TRUE(allClose((a > a), falses)) << "dtype: " << type;
    ASSERT_TRUE(allClose((c > a), trues)) << "dtype: " << type;
    ASSERT_TRUE(allClose((d > a), falseTrues)) << "dtype: " << type;
    ASSERT_TRUE(allClose((a > d), falses)) << "dtype: " << type;
    assertTensorScalarBinop(c, 1, std::greater<>(), trues);
    assertScalarTensorBinop(0, c, std::greater<>(), falses);
    assertTensorScalarBinop(d, 3, std::greater<>(), falseTrues);
    assertScalarTensorBinop(3, d, std::greater<>(), trueFalses);

    ASSERT_TRUE(allClose((a < a), falses)) << "dtype: " << type;
    ASSERT_TRUE(allClose((c < a), falses)) << "dtype: " << type;
    ASSERT_TRUE(allClose((d < a), falses)) << "dtype: " << type;
    ASSERT_TRUE(allClose((a < d), falseTrues)) << "dtype: " << type;
    assertTensorScalarBinop(c, 1, std::less<>(), falses);
    assertScalarTensorBinop(0, c, std::less<>(), trues);
    assertTensorScalarBinop(d, 3, std::less<>(), trueFalses);
    assertScalarTensorBinop(3, d, std::less<>(), falseTrues);

    ASSERT_TRUE(allClose((a >= a), trues)) << "dtype: " << type;
    ASSERT_TRUE(allClose((c >= a), trues)) << "dtype: " << type;
    ASSERT_TRUE(allClose((d >= a), trues)) << "dtype: " << type;
    ASSERT_TRUE(allClose((a >= d), trueFalses)) << "dtype: " << type;
    assertTensorScalarBinop(c, 2, std::greater_equal<>(), trues);
    assertScalarTensorBinop(1, c, std::greater_equal<>(), falses);
    assertTensorScalarBinop(d, 3, std::greater_equal<>(), falseTrues);
    assertScalarTensorBinop(3, d, std::greater_equal<>(), trueFalses);

    ASSERT_TRUE(allClose((a <= a), trues)) << "dtype: " << type;
    ASSERT_TRUE(allClose((c <= a), falses)) << "dtype: " << type;
    ASSERT_TRUE(allClose((d <= a), trueFalses)) << "dtype: " << type;
    ASSERT_TRUE(allClose((a <= d), trues)) << "dtype: " << type;
    assertTensorScalarBinop(c, 1, std::less_equal<>(), falses);
    assertScalarTensorBinop(2, c, std::less_equal<>(), trues);
    assertTensorScalarBinop(d, 3, std::less_equal<>(), trueFalses);
    assertScalarTensorBinop(3, d, std::less_equal<>(), falseTrues);
  };

  applyToAllDtypes(testComparisonBinops);
}

TEST(TensorBinaryOpsTest, LogicalBinaryOperators) {
  auto falses = fl::full({2, 2}, 0, dtype::b8);
  auto trues = fl::full({2, 2}, 1, dtype::b8);
  auto falseTrues =
      Tensor::fromVector<float>({2, 2}, {0, 1, 0, 1}).astype(fl::dtype::b8);

  auto testLogicalBinops = [&](dtype type) {
    auto a = Tensor::fromVector<float>({2, 2}, {0, 1, 0, 3}).astype(type);
    auto b = Tensor::fromVector<float>({2, 2}, {2, 3, 4, 5}).astype(type);
    auto z = fl::full({2, 2}, 0, type);

    ASSERT_TRUE(allClose((z || z), falses)) << "dtype: " << type;
    assertCommutativeBinop(a, z, std::logical_or<>(), falseTrues);
    assertCommutativeBinop(z, b, std::logical_or<>(), trues);
    assertCommutativeBinop(a, b, std::logical_or<>(), trues);
    assertScalarTensorCommutativeBinop(0, a, std::logical_or<>(), falseTrues);
    assertScalarTensorCommutativeBinop(2, z, std::logical_or<>(), trues);

    ASSERT_TRUE(allClose((z && z), falses)) << "dtype: " << type;
    assertCommutativeBinop(a, z, std::logical_and<>(), falses);
    assertCommutativeBinop(z, b, std::logical_and<>(), falses);
    assertCommutativeBinop(a, b, std::logical_and<>(), falseTrues);
    assertScalarTensorCommutativeBinop(0, a, std::logical_and<>(), falses);
    assertScalarTensorCommutativeBinop(2, a, std::logical_and<>(), falseTrues);
  };

  applyToAllDtypes(testLogicalBinops);
}

TEST(TensorBinaryOpsTest, ModuloBinaryOperators) {
  auto testModuloBinop = [](dtype type) {
    auto a = Tensor::fromVector<float>({2, 2}, {1, 2, 3, 4}).astype(type);
    auto b = Tensor::fromVector<float>({2, 2}, {2, 3, 5, 7}).astype(type);
    auto c = Tensor::fromVector<float>({2, 2}, {0, 1, 2, 3}).astype(type);
    auto z = fl::full({2, 2}, 0, type);

    ASSERT_TRUE(allClose((z % b), z)) << "dtype: " << type;
    ASSERT_TRUE(allClose((a % a), z)) << "dtype: " << type;
    ASSERT_TRUE(allClose((a % b), a)) << "dtype: " << type;
    ASSERT_TRUE(allClose((b % a), c)) << "dtype: " << type;

    assertScalarTensorBinop(0, a, std::modulus<>(), z);
    assertScalarTensorBinop(11, a, std::modulus<>(), c);
    assertTensorScalarBinop(a, 1, std::modulus<>(), z);
    assertTensorScalarBinop(a, 5, std::modulus<>(), a);
  };

  applyToAllIntegralDtypes(testModuloBinop);
  // TODO ArrayFire needs software impl for fp16 modulo on CUDA backend;
  // bring this test back when supported.
  // testModuloBinop(dtype::f16);
  testModuloBinop(dtype::f32);
  testModuloBinop(dtype::f64);
}

TEST(TensorBinaryOpsTest, BitBinaryOperators) {
  auto testBitBinops = [](dtype type) {
    auto a = Tensor::fromVector<float>({2, 1}, {0b0001, 0b1000}).astype(type);
    auto b = Tensor::fromVector<float>({2, 1}, {0b0010, 0b0100}).astype(type);
    auto c = Tensor::fromVector<float>({2, 1}, {0b0011, 0b1100}).astype(type);
    auto d = Tensor::fromVector<float>({2, 1}, {0b0110, 0b0110}).astype(type);
    auto e = Tensor::fromVector<float>({2, 1}, {0b1000, 0b0001}).astype(type);
    auto g = Tensor::fromVector<float>({2, 1}, {2, 1}).astype(type);
    auto h = Tensor::fromVector<float>({2, 1}, {0b1000, 0b1000}).astype(type);
    auto z = Tensor::fromVector<float>({2, 1}, {0b0000, 0b0000}).astype(type);

    ASSERT_TRUE(allClose((z & z), z)) << "dtype: " << type;
    assertCommutativeBinop(a, b, std::bit_and<>(), z);
    assertCommutativeBinop(z, b, std::bit_and<>(), z);
    assertCommutativeBinop(d, b, std::bit_and<>(), b);
    assertScalarTensorCommutativeBinop(0b0000, b, std::bit_and<>(), z);
    assertScalarTensorCommutativeBinop(0b0110, b, std::bit_and<>(), b);

    ASSERT_TRUE(allClose((z | z), z)) << "dtype: " << type;
    assertCommutativeBinop(a, z, std::bit_or<>(), a);
    assertCommutativeBinop(z, b, std::bit_or<>(), b);
    assertCommutativeBinop(a, b, std::bit_or<>(), c);
    assertScalarTensorCommutativeBinop(0b0000, b, std::bit_or<>(), b);
    assertScalarTensorCommutativeBinop(0b0110, b, std::bit_or<>(), d);

    ASSERT_TRUE(allClose((z ^ z), z)) << "dtype: " << type;
    assertCommutativeBinop(a, z, std::bit_xor<>(), a);
    assertCommutativeBinop(z, b, std::bit_xor<>(), b);
    assertCommutativeBinop(a, b, std::bit_xor<>(), c);
    assertCommutativeBinop(c, c, std::bit_xor<>(), z);
    assertScalarTensorCommutativeBinop(0b0000, b, std::bit_xor<>(), b);
    assertScalarTensorCommutativeBinop(0b1001, a, std::bit_xor<>(), e);

    // TODO test scalar input (need right/left_shift operator)
    ASSERT_TRUE(allClose((z << z), z)) << "dtype: " << type;
    ASSERT_TRUE(allClose((a << z), a)) << "dtype: " << type;
    ASSERT_TRUE(allClose((z << a), z)) << "dtype: " << type;
    ASSERT_TRUE(allClose((b << g), h)) << "dtype: " << type;

    ASSERT_TRUE(allClose((z >> z), z)) << "dtype: " << type;
    ASSERT_TRUE(allClose((a >> z), a)) << "dtype: " << type;
    ASSERT_TRUE(allClose((z >> a), z)) << "dtype: " << type;
    ASSERT_TRUE(allClose((h >> g), b)) << "dtype: " << type;
  };

  applyToAllIntegralDtypes(testBitBinops);
  // ArrayFire doesn't support bit ops for fps
}

TEST(TensorBinaryOpsTest, BinaryOperatorIncompatibleShapes) {
  auto testTensorIncompatibleShapes = [](dtype type,
                                         const Tensor& lhs,
                                         const Tensor& rhs) {
    ASSERT_THROW(Values(lhs + rhs), std::invalid_argument) << "dtype: " << type;
    ASSERT_THROW(Values(lhs - rhs), std::invalid_argument) << "dtype: " << type;
    ASSERT_THROW(Values(lhs * rhs), std::invalid_argument) << "dtype: " << type;
    ASSERT_THROW(Values(lhs / rhs), std::invalid_argument) << "dtype: " << type;
    ASSERT_THROW(Values(lhs == rhs), std::invalid_argument)
        << "dtype: " << type;
    ASSERT_THROW(Values(lhs != rhs), std::invalid_argument)
        << "dtype: " << type;
    ASSERT_THROW(Values(lhs < rhs), std::invalid_argument) << "dtype: " << type;
    ASSERT_THROW(Values(lhs <= rhs), std::invalid_argument)
        << "dtype: " << type;
    ASSERT_THROW(Values(lhs > rhs), std::invalid_argument) << "dtype: " << type;
    ASSERT_THROW(Values(lhs >= rhs), std::invalid_argument)
        << "dtype: " << type;
    ASSERT_THROW(Values(lhs || rhs), std::invalid_argument)
        << "dtype: " << type;
    ASSERT_THROW(Values(lhs && rhs), std::invalid_argument)
        << "dtype: " << type;
    // TODO ArrayFire needs software impl for fp16 modulo on CUDA backend;
    // bring this test back when supported.
    if (type != dtype::f16) {
      ASSERT_THROW(Values(lhs % rhs), std::invalid_argument)
          << "dtype: " << type;
    }
    // these operators are generally not well-defined for fps
    if (type != dtype::f16 && type != dtype::f32 && type != dtype::f64) {
      ASSERT_THROW(Values(lhs | rhs), std::invalid_argument)
          << "dtype: " << type;
      ASSERT_THROW(Values(lhs ^ rhs), std::invalid_argument)
          << "dtype: " << type;
      ASSERT_THROW(Values(lhs << rhs), std::invalid_argument)
          << "dtype: " << type;
      ASSERT_THROW(Values(lhs >> rhs), std::invalid_argument)
          << "dtype: " << type;
    }
  };

  auto testTensorIncompatibleShapesForType = [&](dtype type) {
    auto a = fl::rand({2, 2}, type);
    auto tooManyAxises = fl::rand({4, 5, 6}, type);
    auto tooFewAxises = fl::rand({3}, type);
    auto diffDim = fl::rand({2, 3}, type);
    testTensorIncompatibleShapes(type, a, tooManyAxises);
    testTensorIncompatibleShapes(type, a, tooFewAxises);
    testTensorIncompatibleShapes(type, a, diffDim);
  };

  applyToAllDtypes(testTensorIncompatibleShapesForType);
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

  for (const auto& funcp : functions) {
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
  InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
