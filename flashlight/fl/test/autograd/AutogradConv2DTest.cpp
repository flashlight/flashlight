/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/test/autograd/AutogradTestUtils.h"

using namespace ::testing;
using namespace fl;

using fl::detail::AutogradTestF16;

TEST(AutogradConv2DTest, Convolve) {
  auto in = Variable(fl::rand({10, 9, 8, 7}, fl::dtype::f32), true);
  auto wt = Variable(fl::rand({4, 3, 8, 6}, fl::dtype::f32), true);
  auto bs = Variable(fl::rand({1, 1, 6, 1}, fl::dtype::f32), true);
  int px = 2, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto benchmarks = std::make_shared<fl::detail::ConvBenchmarks>();
  auto funcConvIn = [&](Variable& input) {
    return conv2d(
        input,
        wt,
        // bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvIn, in, 0.06));
  auto funcConvWt = [&](Variable& weight) {
    return conv2d(
        in,
        weight,
        // bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvWt, wt, 0.06));
  auto funcConvBs = [&](Variable& bias) {
    return conv2d(
        in,
        wt,
        bias,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvBs, bs, 0.03));
}

TEST_F(AutogradTestF16, ConvolveF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  const float scaleFactor = 10.0; // scale the input to prevent grad underflow
  auto in =
      Variable(fl::rand({3, 1, 2, 1}, fl::dtype::f16) * scaleFactor, true);
  auto wt = Variable(fl::rand({2, 1, 2, 1}, fl::dtype::f16), true);
  auto bs = Variable(fl::rand({1, 1, 1, 1}, fl::dtype::f16), true);
  int px = 1, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto benchmarks = std::make_shared<detail::ConvBenchmarks>();
  auto funcConvIn = [&](Variable& input) {
    return conv2d(
        input,
        wt,
        bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvIn, in, 5e-1, 0.1));
  auto funcConvWt = [&](Variable& weight) {
    return conv2d(
        in,
        weight,
        bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvWt, wt, 5e-2, 0.1));
  auto funcConvBs = [&](Variable& bias) {
    return conv2d(
        in,
        wt,
        bias,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1,
        benchmarks);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvBs, bs, 3e-2, 0.1));
}

TEST(AutogradConv2DTest, ConvolveFilterGroups) {
  int channel = 8;
  int groups = 2;
  // w x h x c x b
  auto in = Variable(fl::rand({10, 9, channel, 7}, fl::dtype::f32), true);
  // w x h x in x out
  auto wt =
      Variable(fl::rand({4, 3, channel / groups, 6}, fl::dtype::f32), true);
  auto bs = Variable(fl::rand({1, 1, 6, 1}, fl::dtype::f32), true);

  int px = 2, py = 1;
  int sx = 1, sy = 1;
  int dx = 1, dy = 1;
  auto funcConvIn = [&](Variable& input) {
    return conv2d(input, wt, bs, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvIn, in, 0.06));
  auto funcConvWt = [&](Variable& weight) {
    return conv2d(in, weight, bs, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvWt, wt, 0.05));
  auto foncConvBs = [&](Variable& bias) {
    return conv2d(in, wt, bias, sx, sy, px, py, dx, dy, groups);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(foncConvBs, bs, 0.02));
}

TEST(AutogradConv2DTest, ConvolveDilation) {
  auto in = Variable(fl::rand({10, 9, 8, 7}, fl::dtype::f32), true);
  auto wt = Variable(fl::rand({4, 3, 8, 6}, fl::dtype::f32), true);
  auto bs = Variable(fl::rand({1, 1, 6, 1}, fl::dtype::f32), true);
  int px = 2, py = 1;
  int sx = 1, sy = 1;
  int dx = 2, dy = 1;
  auto funcConvIn = [&](Variable& input) {
    return conv2d(
        input,
        wt,
        bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvIn, in, 0.06));
  auto funcConvWt = [&](Variable& weight) {
    return conv2d(
        in,
        weight,
        bs,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvWt, wt, 0.05));
  auto funcConvBs = [&](Variable& bias) {
    return conv2d(
        in,
        wt,
        bias,
        sx,
        sy,
        px,
        py,
        dx,
        dy,
        /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcConvBs, bs, 0.02));
}

TEST(AutogradConv2DTest, WeightNormConv) {
  auto v = Variable(fl::rand({3, 3, 3, 8}), true);
  auto normDim = {0, 1, 2};
  auto g = Variable(
      norm(v, normDim, /* p = */ 2, /* keepDims = */ true).tensor(), true);
  auto in = Variable(fl::rand({7, 7, 3, 8}) * 2 - 2, true);

  auto funcWeightNormIn = [&](Variable& input) {
    auto w = v *
        tileAs(g / norm(v, normDim, /* p = */ 2, /* keepDims = */ true), v);
    return conv2d(
        input,
        w,
        /* sx */ 1,
        /* sy */ 1,
        /* px */ 0,
        /* py */ 0,
        /* dx */ 1,
        /* dy */ 1,
        /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormIn, in, 3E-1));

  auto funcWeightNormV = [&](Variable& input) {
    auto w = input *
        tileAs(g / norm(input, normDim, /* p = */ 2, /* keepDims = */ true),
               input);
    return conv2d(
        in,
        w,
        /* sx */ 1,
        /* sy */ 1,
        /* px */ 0,
        /* py */ 0,
        /* dx */ 1,
        /* dy */ 1,
        /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormV, v, 2E-1));

  auto funcWeightNormG = [&](Variable& input) {
    auto w = v *
        tileAs(input / norm(v, normDim, /* p = */ 2, /* keepDims = */ true),
               v);
    return conv2d(
        in,
        w,
        /* sx */ 1,
        /* sy */ 1,
        /* px */ 0,
        /* py */ 0,
        /* dx */ 1,
        /* dy */ 1,
        /* groups */ 1);
  };
  ASSERT_TRUE(fl::detail::jacobianTestImpl(funcWeightNormG, g, 2E-1));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
