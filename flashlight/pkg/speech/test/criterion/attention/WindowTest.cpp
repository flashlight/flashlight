/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/flashlight.h"

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/speech/criterion/attention/window.h"

using namespace fl;
using namespace fl::pkg::speech;

TEST(WindowTest, MedianWindow) {
  int inputsteps = 12;
  int batchsize = 4;
  int hiddendim = 16;
  int wl = 2;
  int wr = 3;
  auto inputAttnArray =
      fl::abs(fl::randn({1, wl + wr, batchsize}, fl::dtype::f32));
  auto inputAttn = Variable(
      inputAttnArray /
          fl::tile(
              sum(inputAttnArray, {1}, /* keepDims = */ true),
              {1, inputAttnArray.dim(1)}),
      false);

  MedianWindow window(wl, wr);

  // check initialization
  auto mask0 = window.computeWindow(inputAttn, 0, -1, inputsteps, batchsize);

  auto trueSumMask0 = fl::full({1, inputsteps, batchsize}, 0.0, fl::dtype::f32);
  trueSumMask0(fl::span, fl::range(0, wl + wr), fl::span) = 1.0;

  ASSERT_EQ(mask0.shape(), Shape({1, inputsteps, batchsize}));
  ASSERT_TRUE(allClose(fl::exp(mask0.tensor()), trueSumMask0));

  // check next step
  auto mask1 =
      window.computeWindow(inputAttn, 1, inputsteps, inputsteps, batchsize);
  ASSERT_EQ(mask1.shape(), Shape({1, inputsteps, batchsize}));

  // make sure large window size is handled
  MedianWindow largeWindow(100, 100);
  auto maskLarge = largeWindow.computeWindow(
      inputAttn, 0, inputsteps, inputsteps, batchsize);
  trueSumMask0 = fl::full({1, inputsteps, batchsize}, 0, fl::dtype::f32);
  ASSERT_TRUE(allClose(maskLarge.tensor(), trueSumMask0));
}

TEST(WindowTest, MedianWindowWithPad) {
  int inputsteps = 12;
  int batchsize = 2;
  int wl = 3;
  int wr = 5;
  auto inputAttnArray =
      fl::abs(fl::randn({1, wl + wr, batchsize}, fl::dtype::f32));
  auto inputAttn = Variable(
      inputAttnArray /
          fl::tile(
              sum(inputAttnArray, {1}, /* keepDims = */ true),
              {1, inputAttnArray.dim(1)}),
      false);

  MedianWindow window(wl, wr);
  std::vector<int> inpSzRaw = {1, 2};
  Tensor inpSz = Tensor::fromVector({1, batchsize}, inpSzRaw);
  std::vector<int> tgSzRaw = {1, 2};
  Tensor tgSz = Tensor::fromVector({1, batchsize}, tgSzRaw);

  // check initialization
  auto mask0 = window.computeWindow(
      inputAttn, 0, -1, inputsteps, batchsize, inpSz, tgSz);

  auto trueSumMask0 = fl::full({1, inputsteps, batchsize}, 0.0, fl::dtype::f32);
  trueSumMask0(fl::span, fl::range(0, wl + wr), fl::span) = 1.0;
  trueSumMask0(fl::span, fl::range(inputsteps / 2, inputsteps), 0) = 0.0;

  ASSERT_EQ(mask0.shape(), Shape({1, inputsteps, batchsize}));
  ASSERT_TRUE(allClose(fl::exp(mask0.tensor()), trueSumMask0));

  // check next step
  auto mask2 =
      window.computeWindow(inputAttn, 2, 2, inputsteps, batchsize, inpSz, tgSz);
  ASSERT_EQ(mask2.shape(), Shape({1, inputsteps, batchsize}));
  ASSERT_TRUE(
      fl::countNonzero(
          fl::exp(mask2.tensor())(
              0, fl::range(inputsteps - inputsteps / 2, inputsteps), 0) == 0)
          .scalar<unsigned>() == inputsteps / 2);
  ASSERT_TRUE(
      fl::countNonzero(fl::exp(mask2.tensor())(0, fl::span, 0) == 0)
          .scalar<unsigned>() == inputsteps);
}

TEST(WindowTest, StepWindow) {
  int inputsteps = 100;
  int batchsize = 4;
  int hiddendim = 16;
  int targetlen = 30;
  int sMin = 3, sMax = 15;
  double vMin = 2.3, vMax = 7.5;

  Variable inputAttn; // dummy
  std::vector<int> windowBoundaries(2, 0);

  StepWindow window(sMin, sMax, vMin, vMax);

  // check initialization
  auto mask0 =
      window.computeWindow(inputAttn, 0, inputsteps, inputsteps, batchsize);
  auto trueSumMask0 = fl::full({1, inputsteps, batchsize}, 0.0, fl::dtype::f32);
  windowBoundaries[0] = sMin;
  windowBoundaries[1] = sMax;

  trueSumMask0(
      fl::span, fl::range(windowBoundaries[0], windowBoundaries[1]), fl::span) =
      1.0;

  ASSERT_EQ(mask0.shape(), Shape({1, inputsteps, batchsize}));
  ASSERT_TRUE(allClose(fl::exp(mask0.tensor()), trueSumMask0));

  auto mask1 =
      window.computeWindow(inputAttn, 1, inputsteps, inputsteps, batchsize);
  auto trueSumMask1 = fl::full({1, inputsteps, batchsize}, 0.0, fl::dtype::f32);
  windowBoundaries[0] = static_cast<int>(std::round(sMin + vMin));
  windowBoundaries[1] = static_cast<int>(std::round(sMax + vMax));

  trueSumMask1(
      fl::span, fl::range(windowBoundaries[0], windowBoundaries[1]), fl::span) =
      1.0;

  ASSERT_EQ(mask1.shape(), Shape({1, inputsteps, batchsize}));
  ASSERT_TRUE(allClose(fl::exp(mask1.tensor()), trueSumMask1));

  auto maskLarge =
      window.computeWindow(inputAttn, 1000, inputsteps, inputsteps, batchsize);
  auto trueSumMaskLarge =
      fl::full({1, inputsteps, batchsize}, 0.0, fl::dtype::f32);
  windowBoundaries[0] = static_cast<int>(std::round(inputsteps - vMax));
  windowBoundaries[1] = inputsteps;

  trueSumMaskLarge(
      fl::span, fl::range(windowBoundaries[0], windowBoundaries[1]), fl::span) =
      1.0;

  ASSERT_EQ(maskLarge.shape(), Shape({1, inputsteps, batchsize}));
  ASSERT_TRUE(allClose(fl::exp(maskLarge.tensor()), trueSumMaskLarge));

  auto maskV = window.computeVectorizedWindow(targetlen, inputsteps, batchsize);
  ASSERT_EQ(maskV.shape(), Shape({targetlen, inputsteps, batchsize}));

  std::vector<int> inpSzRaw = {1, 2, 2, 2};
  Tensor inpSz = Tensor::fromVector({1, batchsize}, inpSzRaw);
  std::vector<int> tgSzRaw = {1, 2, 2, 2};
  Tensor tgSz = Tensor::fromVector({1, batchsize}, tgSzRaw);

  auto maskVPad = fl::exp(window
                              .computeVectorizedWindow(
                                  targetlen, inputsteps, batchsize, inpSz, tgSz)
                              .tensor());
  ASSERT_EQ(maskVPad.shape(), Shape({targetlen, inputsteps, batchsize}));
  ASSERT_TRUE(
      fl::countNonzero(
          maskVPad(
              fl::span,
              fl::range(inputsteps - inputsteps / 2, inputsteps),
              0) == 0)
          .scalar<unsigned>() == inputsteps / 2 * targetlen);
  ASSERT_TRUE(
      fl::countNonzero(
          maskVPad(
              fl::range(targetlen - targetlen / 2, targetlen), fl::span, 0) ==
          0)
          .scalar<unsigned>() == targetlen / 2 * inputsteps);
}

TEST(WindowTest, SoftWindow) {
  int inputsteps = 100;
  int batchsize = 4;
  int targetlen = 15;
  int offset = 10;
  double avgRate = 5.2, std = 5.0;

  Variable inputAttn; // dummy
  SoftWindow window(std, avgRate, offset);

  auto mask0 =
      window.computeWindow(inputAttn, 0, inputsteps, inputsteps, batchsize);

  Tensor maxv, maxidx;
  max(maxv, maxidx, mask0.tensor(), 1, /* keepDims = */ true);
  std::vector<int> trueMaxidx(batchsize, offset);

  ASSERT_EQ(mask0.shape(), Shape({1, inputsteps, batchsize}));
  ASSERT_TRUE(allClose(
      maxidx.astype(fl::dtype::s32),
      Tensor::fromVector({1, 1, batchsize}, trueMaxidx, fl::dtype::s32)));

  auto maskV = window.computeVectorizedWindow(targetlen, inputsteps, batchsize);
  ASSERT_EQ(maskV.shape(), Shape({targetlen, inputsteps, batchsize}));

  std::vector<int> inpSzRaw = {1, 2, 2, 2};
  Tensor inpSz = Tensor::fromVector({1, batchsize}, inpSzRaw);
  std::vector<int> tgSzRaw = {1, 2, 2, 2};
  Tensor tgSz = Tensor::fromVector({1, batchsize}, tgSzRaw);

  auto maskVPad = fl::exp(window
                              .computeVectorizedWindow(
                                  targetlen, inputsteps, batchsize, inpSz, tgSz)
                              .tensor());
  ASSERT_EQ(maskVPad.shape(), Shape({targetlen, inputsteps, batchsize}));
  ASSERT_TRUE(
      fl::countNonzero(
          maskVPad(
              fl::span,
              fl::range(inputsteps - inputsteps / 2, inputsteps),
              0) == 0)
          .scalar<unsigned>() == inputsteps / 2 * targetlen);
  ASSERT_TRUE(
      fl::countNonzero(
          maskVPad(
              fl::range(targetlen - targetlen / 2, targetlen), fl::span, 0) ==
          0)
          .scalar<unsigned>() == targetlen / 2 * inputsteps);
}

TEST(WindowTest, SoftPretrainWindow) {
  int inputsteps = 32;
  int targetlen = 8;
  int batchsize = 4;
  double std = 5.0;

  std::vector<unsigned int> peaks = {0, 4, 8, 12, 16, 20, 24, 28};

  Variable inputAttn;
  SoftPretrainWindow window(std);

  // single step
  std::vector<Variable> masks;
  for (int step = 0; step < targetlen; ++step) {
    masks.emplace_back(window.computeWindow(
        inputAttn, step, targetlen, inputsteps, batchsize));
  }
  auto maskS = concatenate(masks, 0);
  Tensor maxv, maxidx;
  max(maxv, maxidx, maskS.tensor()(fl::span, fl::span, 0), 1);

  ASSERT_EQ(maskS.shape(), Shape({targetlen, inputsteps, batchsize}));
  ASSERT_TRUE(allClose(maxidx, Tensor::fromVector({8}, peaks)));

  // vectorized
  auto maskV = window.computeVectorizedWindow(targetlen, inputsteps, batchsize);
  max(maxv, maxidx, maskV.tensor()(fl::span, fl::span, 0), 1);

  ASSERT_EQ(maskV.shape(), Shape({targetlen, inputsteps, batchsize}));
  ASSERT_TRUE(allClose(maxidx, Tensor::fromVector({8}, peaks)));
  ASSERT_TRUE(allClose(maskS, maskV));

  std::vector<int> inpSzRaw = {1, 2, 2, 2};
  Tensor inpSz = Tensor::fromVector({1, batchsize}, inpSzRaw);
  std::vector<int> tgSzRaw = {1, 2, 2, 2};
  Tensor tgSz = Tensor::fromVector({1, batchsize}, tgSzRaw);

  auto maskVPad = fl::exp(window
                              .computeVectorizedWindow(
                                  targetlen, inputsteps, batchsize, inpSz, tgSz)
                              .tensor());
  ASSERT_EQ(maskVPad.shape(), Shape({targetlen, inputsteps, batchsize}));
  ASSERT_TRUE(
      fl::countNonzero(
          maskVPad(
              fl::span,
              fl::range(inputsteps - inputsteps / 2, inputsteps),
              0) == 0)
          .scalar<unsigned>() == inputsteps / 2 * targetlen);
  ASSERT_TRUE(
      fl::countNonzero(
          maskVPad(
              fl::range(targetlen - targetlen / 2, targetlen), fl::span, 0) ==
          0)
          .scalar<unsigned>() == targetlen / 2 * inputsteps);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
