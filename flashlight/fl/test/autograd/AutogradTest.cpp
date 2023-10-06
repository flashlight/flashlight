/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace fl;

using fl::detail::AutogradTestF16;

TEST(AutogradTest, OperatorParenthesis) {
  auto x = Variable(fl::rand({1, 3, 3}, fl::dtype::f64), true);
  auto y = x(0, 0) + x(0, 1);
  auto funcOperatorParen = [](Variable& in) { return in(0, 0) + in(0, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcOperatorParen, x));
}

TEST(AutogradTest, AutogradOperatorTypeCompatibility) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto f16 = Variable(fl::rand({2, 2}, fl::dtype::f16), true);
  auto f32 = Variable(fl::rand({2, 2}, fl::dtype::f32), true);

  // Binary operators
  EXPECT_THROW({ auto res = f16 + f32; }, std::invalid_argument); // +
  EXPECT_THROW({ auto res = f16 - f32; }, std::invalid_argument); // -
  EXPECT_THROW({ auto res = f16 * f32; }, std::invalid_argument); // *
  EXPECT_THROW({ auto res = f16 / f32; }, std::invalid_argument); // /
  EXPECT_THROW({ auto res = f16 > f32; }, std::invalid_argument); // >
  EXPECT_THROW({ auto res = f16 < f32; }, std::invalid_argument); // <
  EXPECT_THROW({ auto res = f16 >= f32; }, std::invalid_argument); // >=
  EXPECT_THROW({ auto res = f16 <= f32; }, std::invalid_argument); // <=
  EXPECT_THROW({ auto res = f16 && f32; }, std::invalid_argument); // &&
  EXPECT_THROW({ max(f16, f32); }, std::invalid_argument); // max
  EXPECT_THROW({ min(f16, f32); }, std::invalid_argument); // min
  EXPECT_THROW({ matmul(f16, f32); }, std::invalid_argument); // matmul
  EXPECT_THROW({ matmulTN(f16, f32); }, std::invalid_argument); // matmulTN
  EXPECT_THROW({ matmulNT(f16, f32); }, std::invalid_argument); // matmulNT
  EXPECT_NO_THROW({ binaryCrossEntropy(f16, f32); });
  EXPECT_NO_THROW({
    categoricalCrossEntropy(
        Variable(fl::rand({7, 10, 4}, fl::dtype::f16), true),
        Variable(
            (fl::rand({10, 4}, fl::dtype::u32) % 7).astype(fl::dtype::s32),
            false));
  });
  EXPECT_NO_THROW({ pool2d(f16, 1, 1, 1, 1, 1, 1); });
  EXPECT_NO_THROW({ embedding(f16, f32); }); // lookup is of a different type
  // Ternary operators
  auto f32_2 = Variable(fl::rand({2, 2}, fl::dtype::f32), true);
  auto f16_2 = Variable(fl::rand({2, 2}, fl::dtype::f16), true);
  EXPECT_THROW({ linear(f16, f32, f16_2); }, std::invalid_argument); // linear
  EXPECT_THROW({ linear(f16, f32, f32_2); }, std::invalid_argument); // linear
  auto w = Variable(fl::rand({1}, fl::dtype::f32), true);
  auto b = Variable(fl::rand({1}, fl::dtype::f32), true);
  EXPECT_THROW(
      { batchnorm(f16, f32, f32_2, w, b, {1}, true, 0.01, 0.01); },
      std::invalid_argument);
  EXPECT_THROW(
      { batchnorm(f16, f32, f16_2, w, b, {1}, true, 0.01, 0.01); },
      std::invalid_argument);
  EXPECT_THROW(
      { conv2d(f16, f32, f16_2, 1, 1, 0, 0, 1, 1); }, std::invalid_argument);
  // Quaternary
  auto f16_3 = Variable(fl::rand({2, 2, 3}, fl::dtype::f16), false);
  auto f16_4 = Variable(fl::rand({50}, fl::dtype::f16), false);
  EXPECT_THROW(
      {
        rnn(f16_3,
            Variable(Tensor(fl::dtype::f32), false),
            Variable(Tensor(fl::dtype::f32), false),
            f16_4,
            2,
            2,
            RnnMode::LSTM,
            true,
            0.0);
      },
      std::invalid_argument);
  // Variadic operators
  std::vector<Variable> concatInputs = {f16, f32, f16_2, f32_2};
  EXPECT_THROW({ concatenate(concatInputs, 0); }, std::invalid_argument);
}

TEST(AutogradTest, CastingAsDifferentGradTypes) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto f32 = Variable(fl::rand({5, 5}), true);
  auto f16 = Variable(fl::rand({5, 5}, fl::dtype::f16), true);
  // Computing gradients with mixed types fails when the op is applied
  ASSERT_THROW({ f32 + f16; }, std::invalid_argument);
}

TEST(AutogradTest, CastingAs) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto var = Variable(fl::rand({5, 5}), true);
  auto varF16 = var.astype(fl::dtype::f16);
  ASSERT_EQ(var.type(), fl::dtype::f32);
  ASSERT_EQ(varF16.type(), fl::dtype::f16);
  ASSERT_TRUE(allClose(varF16.tensor(), var.astype(fl::dtype::f16).tensor()));
}

TEST(AutogradTest, CastingAsBackward) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto a = Variable(fl::rand({4, 4}, fl::dtype::f16), true);
  auto b = Variable(fl::rand({4, 4}, fl::dtype::f16), false);
  auto c = b + a;
  c.backward();
  ASSERT_EQ(a.grad().type(), fl::dtype::f16);
  ASSERT_EQ(a.grad().type(), fl::dtype::f16);
  a = a.astype(fl::dtype::f32);
  ASSERT_FALSE(a.isGradAvailable());
}

TEST(AutogradTest, CastingAsGrad) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // compare to f32 case
  auto x = Variable(fl::full({5}, 2.0), true);
  auto y = Variable(fl::full({5}, 3.0), true);
  auto z = x * x + x * y + y * y;
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  auto dy = y.grad();

  // f16 -- cast gradients in both directions
  auto x32 = Variable(fl::full({5}, 2.0), true);
  auto y32 = Variable(fl::full({5}, 3.0), true);
  auto xf16 = x32.astype(fl::dtype::f16);
  auto yf16 = y32.astype(fl::dtype::f16);
  auto zf16 = xf16 * xf16 + xf16 * yf16 + yf16 * yf16;
  auto zf32 = zf16.astype(fl::dtype::f32);
  zf32.backward(dz);

  ASSERT_EQ(xf16.grad().type(), fl::dtype::f16);
  ASSERT_EQ(yf16.grad().type(), fl::dtype::f16);
  ASSERT_EQ(zf16.grad().type(), fl::dtype::f16);
  ASSERT_EQ(x32.grad().type(), fl::dtype::f32);
  ASSERT_EQ(y32.grad().type(), fl::dtype::f32);
  ASSERT_TRUE(
      allClose(dx.tensor(), xf16.grad().tensor().astype(fl::dtype::f32)));
  ASSERT_TRUE(
      allClose(dy.tensor(), y32.grad().tensor().astype(fl::dtype::f32)));
  ASSERT_TRUE(allClose(dx.tensor(), x32.grad().tensor()));
  ASSERT_TRUE(allClose(dy.tensor(), y32.grad().tensor()));
}

TEST(AutogradTest, NoCalcGrad) {
  auto x = Variable(fl::rand({5}), false);
  auto y = Variable(fl::rand({5}), true);
  auto z = x * x + x * y + y * y;
  auto dz = Variable(fl::full({5}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  ASSERT_TRUE(allClose(dy.tensor(), 2 * y.tensor() + x.tensor()));
  ASSERT_THROW(x.grad(), std::logic_error);
}

TEST(AutogradTest, Concatenate) {
  auto x1 = Variable(fl::rand({2, 3, 1, 2}, fl::dtype::f64), true);
  auto x2 = Variable(fl::rand({2, 3, 3, 2}, fl::dtype::f64), true);
  auto x3 = Variable(fl::rand({2, 3, 1, 2}, fl::dtype::f64), true);
  auto x4 = Variable(fl::rand({2, 3, 7, 2}, fl::dtype::f64), true);
  std::vector<Variable> inputs = {x1, x2, x3, x4};
  auto output = concatenate(inputs, 2);

  ASSERT_EQ(output.shape(), Shape({2, 3, 12, 2}));

  auto funcConcatenateT1 = [x2, x3, x4](Variable& in) {
    return concatenate({in, x2, x3, x4}, 2);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConcatenateT1, x1));

  auto funcConcatenateT2 = [x1, x2, x4](Variable& in) {
    return concatenate({x1, x2, in, x4}, 2);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConcatenateT2, x3));
}

TEST(AutogradTest, Split) {
  // check output
  auto x = Variable(fl::arange({7, 2}), true);
  auto yVec = split(x, 1, 0);
  ASSERT_EQ(yVec.size(), 7);
  ASSERT_EQ(yVec[0].shape(), Shape({1, 2}));
  ASSERT_EQ(yVec[2].shape(), Shape({1, 2}));
  ASSERT_TRUE(fl::all(yVec[6].tensor() == 6).scalar<char>());

  auto a = Variable(fl::arange({5, 3}, 1), true);
  auto bVec = split(a, {2, 1}, 1);
  ASSERT_EQ(bVec.size(), 2);
  ASSERT_EQ(bVec[0].shape(), Shape({5, 2}));
  ASSERT_EQ(bVec[1].shape(), Shape({5, 1}));
  ASSERT_TRUE(
      fl::all(bVec[0].tensor() == fl::arange({5, 2}, 1)).scalar<char>());
  ASSERT_TRUE(fl::all(bVec[1].tensor() == 2).scalar<char>());

  // check exception handling
  ASSERT_THROW(split(a, {2, 2}, 0), std::invalid_argument);

  // check gradient
  auto gradFunc = [](Variable& in) { return split(in, 2, 1)[0]; };
  auto input = Variable(fl::rand({2, 3}, fl::dtype::f64), true);
  ASSERT_TRUE(fl::detail::jacobianTestImpl(gradFunc, input));
}

TEST(AutogradTest, Tile) {
  auto x = Variable(fl::rand({6}), true);
  auto y = Variable(fl::rand({6, 3}), true);
  auto z = y * tile(x, {1, 3});
  auto dz = Variable(fl::full({6, 3}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.tensor(), fl::tile(x.tensor(), {1, 3})));
  ASSERT_TRUE(allClose(dx.tensor(), fl::sum(y.tensor(), {1})));

  // Jacobian
  auto input = Variable(fl::rand({10, 1, 5}), true);
  auto funcTile = [](Variable& in) { return tile(in, {1, 2}); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcTile, input, 1E-4, 1E-3));
}

TEST(AutogradTest, TileAs) {
  auto x = Variable(fl::rand({5}), true);
  auto y = Variable(fl::rand({5, 2}), true);
  auto z = y * tileAs(x, y);
  auto dz = Variable(fl::full({5, 2}, 1.0), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dy.tensor(), fl::tile(x.tensor(), {1, 2})));
  ASSERT_TRUE(allClose(dx.tensor(), fl::sum(y.tensor(), {1})));
}

TEST_F(AutogradTestF16, TileAsF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto x = Variable(fl::rand({5}, fl::dtype::f16), true);
  auto y = Variable(fl::rand({5, 2}, fl::dtype::f16), true);
  auto z = y * tileAs(x, y);
  ASSERT_EQ(x.type(), z.type());
  auto dz = Variable(fl::full({5, 2}, 1.0, fl::dtype::f16), false);
  z.backward(dz);
  auto dy = y.grad();
  auto dx = x.grad();
  ASSERT_TRUE(allClose(
      dy.tensor(), fl::tile(x.tensor(), {1, 2}).astype(dx.type()), 1e-2));
  ASSERT_TRUE(
      allClose(dx.tensor(), fl::sum(y.tensor(), {1}).astype(dx.type()), 1e-2));
}

TEST(AutogradTest, TileAs2) {
  auto x = Variable(fl::rand({10}), true);
  auto z = tileAs(x, Shape({10, 3}));
  auto dz = Variable(fl::full({10, 3}, 1.0), false);
  z.backward(dz);
  auto dx = x.grad();
  ASSERT_TRUE(allClose(dx.tensor(), fl::full(x.shape(), 3.0)));
}

TEST(AutogradTest, Indexing) {
  auto x = Variable(fl::rand({5, 6, 7, 4}, fl::dtype::f64), true);

  auto funcCol = [](Variable& input) { return input(fl::span, 4); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcCol, x));

  auto funcRow = [](Variable& input) { return input(4); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcRow, x));

  auto funcSlice = [](Variable& input) {
    return input(fl::span, fl::span, 4);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSlice, x));

  auto funcCols = [](Variable& input) {
    return input(fl::span, fl::range(2, 5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcCols, x));

  auto funcRows = [](Variable& input) { return input(fl::range(2, 5)); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcRows, x));

  auto funcSlices = [](Variable& input) {
    return input(fl::span, fl::span, fl::range(2, 5));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcSlices, x));
  auto funcFlat = [](Variable& input) {
    return input.flat(fl::range(4, 100));
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcFlat, x));
}

TEST(AutogradTest, Padding) {
  auto in = Variable(fl::rand({3, 3}, fl::dtype::f32), true);
  auto funcPad = [&](Variable& input) {
    return padding(input, {{1, 2}, {0, 1}}, -1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcPad, in, 1E-3));
}

TEST(AutogradTest, Pooling) {
  auto in = Variable(fl::rand({3, 3, 1, 1}, fl::dtype::f32), true);
  auto funcPool = [&](Variable& input) { return pool2d(input, 2, 2, 1, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcPool, in, 1E-3));
}

TEST_F(AutogradTestF16, PoolingF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  const float inputScale = 2.0; // scale the input to prevent grad underflow
  auto in = Variable(inputScale * fl::rand({3, 3, 1, 1}, fl::dtype::f16), true);
  auto funcPool = [&](Variable& input) { return pool2d(input, 2, 2, 1, 1); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcPool, in, 1e1, 1e-1)); // TODO: investigate
}

TEST(AutogradTest, Reorder) {
  auto in = Variable(fl::rand({3, 1, 4, 1}, fl::dtype::f32) * 2, true);
  auto funcReorder = [&](Variable& input) {
    return reorder(input, {2, 0, 3, 1});
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcReorder, in, 1E-3));
}

TEST(AutogradTest, Embedding) {
  int nWords = 10;
  auto input =
      Variable((fl::rand({4, 2}) * nWords).astype(fl::dtype::f32), false);
  auto weights = Variable(fl::randn({4, nWords}, fl::dtype::f64), true);
  auto funcEmbed = [&](Variable& w) { return embedding(input, w); };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcEmbed, weights, 1E-5));
}

TEST(AutogradTest, GetAdvancedIndex) {
  // TODO: remove me
  if (!FL_BACKEND_CUDA) {
    GTEST_SKIP()
        << "Advanced indexing operator unsupported for non-CUDA backends";
  }
  std::vector<fl::dtype> validIndexTypes = {
      fl::dtype::s32, fl::dtype::s64, fl::dtype::u32, fl::dtype::u64};
  for (const auto& dtype : validIndexTypes) {
    auto x = Variable(fl::rand({20, 50, 40, 30}, fl::dtype::f32), true);
    Tensor a({6}, dtype);
    a(0) = 0;
    a(1) = 15;
    a(2) = 6;
    a(3) = 1;
    a(4) = 10;
    a(5) = 6;
    Tensor b({3}, dtype);
    b(0) = 5;
    b(1) = 11;
    b(2) = 19;
    auto x2 = x(a, b, fl::span, fl::range(0, 4));
    auto y = sum(x2 * x2, {0, 1, 2, 3}, /* keepDims = */ true);
    auto res = 2 * sum(x2, {0, 1, 2, 3}, /* keepDims = */ true).tensor();
    y.backward();
    auto grad = sum(x.grad(), {0, 1, 2, 3}, /* keepDims = */ true).tensor();
    ASSERT_TRUE(allClose(grad, res, 1e-3));
  }
}

TEST(AutogradTest, GetAdvancedIndexF16) {
  // TODO: remove me
  if (!FL_BACKEND_CUDA) {
    GTEST_SKIP()
        << "Advanced indexing operator unsupported for non-CUDA backends";
  }
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }
  std::vector<fl::dtype> validIndexTypes = {
      fl::dtype::s32, fl::dtype::s64, fl::dtype::u32, fl::dtype::u64};
  for (const auto& dtype : validIndexTypes) {
    auto x = Variable(fl::rand({20, 50, 40, 30}, fl::dtype::f16), true);
    Tensor a({6}, dtype);
    a(0) = 0;
    a(1) = 15;
    a(2) = 6;
    a(3) = 1;
    a(4) = 10;
    a(5) = 6;
    Tensor b({3}, dtype);
    b(0) = 5;
    b(1) = 11;
    b(2) = 19;
    auto x2 = x(a, b, fl::span, fl::range(0, 4));
    ASSERT_EQ(x2.type(), fl::dtype::f16);
    auto y = sum(x2 * x2, {0, 1, 2, 3}, /* keepDims = */ true);
    auto res = 2 * sum(x2, {0, 1, 2, 3}, /* keepDims = */ true).tensor();
    y.backward();
    ASSERT_EQ(x.grad().type(), fl::dtype::f16);
    auto grad = sum(x.grad(), {0, 1, 2, 3}, /* keepDims = */ true).tensor();
    ASSERT_TRUE(allClose(grad, res, 1e-3));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
