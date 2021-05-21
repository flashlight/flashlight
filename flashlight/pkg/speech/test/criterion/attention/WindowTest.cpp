/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <arrayfire.h>
#include "flashlight/fl/flashlight.h"

#include "flashlight/pkg/speech/criterion/attention/window.h"

using namespace fl;
using namespace fl::app::asr;

TEST(WindowTest, MedianWindow) {
  int inputsteps = 12;
  int batchsize = 4;
  int hiddendim = 16;
  int wl = 2;
  int wr = 3;
  auto inputAttnArray = af::abs(af::randn(1, wl + wr, batchsize, f32));
  auto inputAttn = Variable(
      inputAttnArray /
          af::tile(sum(inputAttnArray, 1), 1, inputAttnArray.dims(1)),
      false);

  MedianWindow window(wl, wr);

  // check initialization
  auto mask0 = window.computeWindow(inputAttn, 0, -1, inputsteps, batchsize);

  auto trueSumMask0 = af::constant(0.0, 1, inputsteps, batchsize, f32);
  trueSumMask0(af::span, af::seq(0, wl + wr - 1), af::span) = 1.0;

  ASSERT_EQ(mask0.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(af::exp(mask0.array()), trueSumMask0));

  // check next step
  auto mask1 = window.computeWindow(inputAttn, 1, inputsteps, inputsteps, batchsize);
  ASSERT_EQ(mask1.dims(), af::dim4(1, inputsteps, batchsize));

  // make sure large window size is handled
  MedianWindow largeWindow(100, 100);
  auto maskLarge =
      largeWindow.computeWindow(inputAttn, 0, inputsteps, inputsteps, batchsize);
  trueSumMask0 = af::constant(0, 1, inputsteps, batchsize, f32);
  ASSERT_TRUE(allClose(maskLarge.array(), trueSumMask0));
}

TEST(WindowTest, MedianWindowWithPad) {
  int inputsteps = 12;
  int batchsize = 2;
  int wl = 3;
  int wr = 5;
  auto inputAttnArray = af::abs(af::randn(1, wl + wr, batchsize, f32));
  auto inputAttn = Variable(
      inputAttnArray /
          af::tile(sum(inputAttnArray, 1), 1, inputAttnArray.dims(1)),
      false);

  MedianWindow window(wl, wr);
  std::vector<int> inpSzRaw = {1, 2};
  af::array inpSz = af::array(af::dim4(1, batchsize), inpSzRaw.data());
  std::vector<int> tgSzRaw = {1, 2};
  af::array tgSz = af::array(af::dim4(1, batchsize), tgSzRaw.data());

  // check initialization
  auto mask0 = window.computeWindow(
      inputAttn, 0, -1, inputsteps, batchsize, inpSz, tgSz);

  auto trueSumMask0 = af::constant(0.0, 1, inputsteps, batchsize, f32);
  trueSumMask0(af::span, af::seq(0, wl + wr - 1), af::span) = 1.0;
  trueSumMask0(af::span, af::seq(inputsteps / 2, inputsteps - 1), 0) = 0.0;

  ASSERT_EQ(mask0.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(af::exp(mask0.array()), trueSumMask0));

  // check next step
  auto mask2 =
      window.computeWindow(inputAttn, 2, 2, inputsteps, batchsize, inpSz, tgSz);
  ASSERT_EQ(mask2.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(
      af::count<int>(
          af::exp(mask2.array())(
              0, af::seq(inputsteps - inputsteps / 2, inputsteps - 1), 0) ==
          0) == inputsteps / 2);
  ASSERT_TRUE(
      af::count<int>(af::exp(mask2.array())(0, af::span, 0) == 0) ==
      inputsteps);
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
  auto mask0 = window.computeWindow(inputAttn, 0, inputsteps, inputsteps, batchsize);
  auto trueSumMask0 = af::constant(0.0, 1, inputsteps, batchsize, f32);
  windowBoundaries[0] = sMin;
  windowBoundaries[1] = sMax;

  trueSumMask0(
      af::span,
      af::seq(windowBoundaries[0], windowBoundaries[1] - 1),
      af::span) = 1.0;

  ASSERT_EQ(mask0.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(af::exp(mask0.array()), trueSumMask0));

  auto mask1 = window.computeWindow(inputAttn, 1, inputsteps, inputsteps, batchsize);
  auto trueSumMask1 = af::constant(0.0, 1, inputsteps, batchsize, f32);
  windowBoundaries[0] = static_cast<int>(std::round(sMin + vMin));
  windowBoundaries[1] = static_cast<int>(std::round(sMax + vMax));

  trueSumMask1(
      af::span,
      af::seq(windowBoundaries[0], windowBoundaries[1] - 1),
      af::span) = 1.0;

  ASSERT_EQ(mask1.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(af::exp(mask1.array()), trueSumMask1));

  auto maskLarge =
      window.computeWindow(inputAttn, 1000, inputsteps, inputsteps, batchsize);
  auto trueSumMaskLarge = af::constant(0.0, 1, inputsteps, batchsize, f32);
  windowBoundaries[0] = static_cast<int>(std::round(inputsteps - vMax));
  windowBoundaries[1] = inputsteps;

  trueSumMaskLarge(
      af::span,
      af::seq(windowBoundaries[0], windowBoundaries[1] - 1),
      af::span) = 1.0;

  ASSERT_EQ(maskLarge.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(af::exp(maskLarge.array()), trueSumMaskLarge));

  auto maskV = window.computeVectorizedWindow(targetlen, inputsteps, batchsize);
  ASSERT_EQ(maskV.dims(), af::dim4(targetlen, inputsteps, batchsize));

  std::vector<int> inpSzRaw = {1, 2, 2, 2};
  af::array inpSz = af::array(af::dim4(1, batchsize), inpSzRaw.data());
  std::vector<int> tgSzRaw = {1, 2, 2, 2};
  af::array tgSz = af::array(af::dim4(1, batchsize), tgSzRaw.data());

  auto maskVPad = af::exp(window
                              .computeVectorizedWindow(
                                  targetlen, inputsteps, batchsize, inpSz, tgSz)
                              .array());
  ASSERT_EQ(maskVPad.dims(), af::dim4(targetlen, inputsteps, batchsize));
  ASSERT_TRUE(
      af::count<int>(
          maskVPad(
              af::span,
              af::seq(inputsteps - inputsteps / 2, inputsteps - 1),
              0) == 0) == inputsteps / 2 * targetlen);
  ASSERT_TRUE(
      af::count<int>(
          maskVPad(
              af::seq(targetlen - targetlen / 2, targetlen - 1), af::span, 0) ==
          0) == targetlen / 2 * inputsteps);
}

TEST(WindowTest, SoftWindow) {
  int inputsteps = 100;
  int batchsize = 4;
  int targetlen = 15;
  int offset = 10;
  double avgRate = 5.2, std = 5.0;

  Variable inputAttn; // dummy
  SoftWindow window(std, avgRate, offset);

  auto mask0 = window.computeWindow(inputAttn, 0, inputsteps, inputsteps, batchsize);

  af::array maxv, maxidx;
  max(maxv, maxidx, mask0.array(), 1);
  std::vector<int> trueMaxidx(batchsize, offset);

  ASSERT_EQ(mask0.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(
      maxidx.as(af::dtype::s32),
      af::array(1, 1, batchsize, trueMaxidx.data())));

  auto maskV = window.computeVectorizedWindow(targetlen, inputsteps, batchsize);
  ASSERT_EQ(maskV.dims(), af::dim4(targetlen, inputsteps, batchsize));

  std::vector<int> inpSzRaw = {1, 2, 2, 2};
  af::array inpSz = af::array(af::dim4(1, batchsize), inpSzRaw.data());
  std::vector<int> tgSzRaw = {1, 2, 2, 2};
  af::array tgSz = af::array(af::dim4(1, batchsize), tgSzRaw.data());

  auto maskVPad = af::exp(window
                              .computeVectorizedWindow(
                                  targetlen, inputsteps, batchsize, inpSz, tgSz)
                              .array());
  ASSERT_EQ(maskVPad.dims(), af::dim4(targetlen, inputsteps, batchsize));
  ASSERT_TRUE(
      af::count<int>(
          maskVPad(
              af::span,
              af::seq(inputsteps - inputsteps / 2, inputsteps - 1),
              0) == 0) == inputsteps / 2 * targetlen);
  ASSERT_TRUE(
      af::count<int>(
          maskVPad(
              af::seq(targetlen - targetlen / 2, targetlen - 1), af::span, 0) ==
          0) == targetlen / 2 * inputsteps);
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
  af::array maxv, maxidx;
  max(maxv, maxidx, maskS.array()(af::span, af::span, 0), 1);

  ASSERT_EQ(maskS.dims(), af::dim4(targetlen, inputsteps, batchsize));
  ASSERT_TRUE(allClose(maxidx, af::array(8, peaks.data())));

  // vectorized
  auto maskV = window.computeVectorizedWindow(targetlen, inputsteps, batchsize);
  max(maxv, maxidx, maskV.array()(af::span, af::span, 0), 1);

  ASSERT_EQ(maskV.dims(), af::dim4(targetlen, inputsteps, batchsize));
  ASSERT_TRUE(allClose(maxidx, af::array(8, peaks.data())));
  ASSERT_TRUE(allClose(maskS, maskV));

  std::vector<int> inpSzRaw = {1, 2, 2, 2};
  af::array inpSz = af::array(af::dim4(1, batchsize), inpSzRaw.data());
  std::vector<int> tgSzRaw = {1, 2, 2, 2};
  af::array tgSz = af::array(af::dim4(1, batchsize), tgSzRaw.data());

  auto maskVPad = af::exp(window
                              .computeVectorizedWindow(
                                  targetlen, inputsteps, batchsize, inpSz, tgSz)
                              .array());
  ASSERT_EQ(maskVPad.dims(), af::dim4(targetlen, inputsteps, batchsize));
  ASSERT_TRUE(
      af::count<int>(
          maskVPad(
              af::span,
              af::seq(inputsteps - inputsteps / 2, inputsteps - 1),
              0) == 0) == inputsteps / 2 * targetlen);
  ASSERT_TRUE(
      af::count<int>(
          maskVPad(
              af::seq(targetlen - targetlen / 2, targetlen - 1), af::span, 0) ==
          0) == targetlen / 2 * inputsteps);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
