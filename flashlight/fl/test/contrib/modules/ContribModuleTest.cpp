/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/nn/nn.h"

using namespace fl;

namespace {

class ModuleTestF16 : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensures all operations will be in f16
    OptimMode::get().setOptimLevel(OptimLevel::O3);
  }

  void TearDown() override {
    OptimMode::get().setOptimLevel(OptimLevel::DEFAULT);
  }
};

} // namespace

TEST(ModuleTest, ResidualFwd) {
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2);
  auto bn = BatchNorm(2, 50);
  auto relu = ReLU();

  int batchsize = 10;
  auto input = Variable(af::randu(120, 100, 30, batchsize), false);

  auto outputConv = conv.forward(input);
  auto outputBn = bn.forward(outputConv);

  auto resModule1 = Residual();
  resModule1.add(conv);
  resModule1.add(bn);
  resModule1.add(relu);
  resModule1.addShortcut(1, 3);

  auto output1 = resModule1.forward(input);
  auto output1True = relu.forward(outputBn + outputConv);
  ASSERT_TRUE(allClose(output1, output1True));

  auto resModule2 = Residual();
  resModule2.add(conv);
  resModule2.add(bn);
  resModule2.add(relu);
  resModule2.addShortcut(1, 4);
  resModule2.addShortcut(1, 3);
  resModule2.addShortcut(2, 4);

  auto output2 = resModule2.forward(input);
  auto output2True =
      relu.forward(outputBn + outputConv) + outputBn + outputConv;
  ASSERT_TRUE(allClose(output2, output2True));
}

TEST(ModuleTest, ResidualFwdWithProjection) {
  const float proj1FwdScale = 0.24;
  const float proj2FwdScale = 0.5;
  const float linFwdScale = 0.3;

  auto linear1 = Linear(12, 8);
  auto relu1 = ReLU();
  auto linear2 = Linear(8, 4);
  auto relu2 = ReLU();
  auto linear3 = Linear(4, 4);
  auto relu3 = ReLU();
  auto projection1 = Linear(8, 4);
  auto projection2 = Linear(12, 4);

  auto input = Variable(af::randu(12, 10, 3, 4), false);
  auto output1True = linear1.forward(input);
  auto outputTrue = relu1.forward(output1True);
  outputTrue = linear2.forward(outputTrue * linFwdScale);
  outputTrue = relu2.forward(
      (outputTrue + projection1.forward(output1True)) * proj1FwdScale);
  outputTrue = (outputTrue + projection2.forward(input)) * proj2FwdScale;
  outputTrue = linear3.forward(outputTrue);
  outputTrue = relu3.forward(outputTrue) + outputTrue;

  auto resModule = Residual();
  resModule.add(linear1);
  resModule.add(relu1);
  resModule.add(linear2);
  resModule.addScale(3, linFwdScale);
  resModule.add(relu2);
  resModule.addShortcut(1, 4, projection1);
  resModule.addScale(4, proj1FwdScale);
  resModule.add(linear3);
  resModule.addShortcut(0, 5, projection2);
  resModule.addScale(5, proj2FwdScale);
  resModule.add(relu3);
  resModule.addShortcut(5, 7);

  auto outputRes = resModule.forward(input);
  ASSERT_TRUE(allClose(outputRes, outputTrue));
}

TEST(ModuleTest, AsymmetricConv1DFwd) {
  int batchsize = 10;
  int timesteps = 120;
  int c = 32;

  auto conv = AsymmetricConv1D(c, c, 5, 1, -1, 0, 1); // use only past
  auto input = Variable(af::randu(timesteps, 1, c, batchsize), false);

  auto output = conv.forward(input);

  ASSERT_EQ(output.dims(0), timesteps);
  ASSERT_EQ(output.dims(1), 1);
  ASSERT_EQ(output.dims(2), c);

  auto convFuture = AsymmetricConv1D(c, c, 5, 1, -1, 1, 1); // use only future
  auto outputFuture = convFuture.forward(input);
  ASSERT_EQ(outputFuture.dims(0), timesteps);
  ASSERT_EQ(outputFuture.dims(1), 1);
  ASSERT_EQ(outputFuture.dims(2), c);

  ASSERT_FALSE(allClose(output, outputFuture));
}

TEST(ModuleTest, TransformerFwd) {
  int batchsize = 10;
  int timesteps = 120;
  int c = 32;
  int nheads = 4;

  auto tr =
      Transformer(c, c / nheads, c, nheads, timesteps, 0.2, 0.1, false, false);
  auto input = Variable(af::randu(c, timesteps, batchsize, 1), false);

  auto output = tr.forward({input});

  ASSERT_EQ(output[0].dims(0), c);
  ASSERT_EQ(output[0].dims(1), timesteps);
  ASSERT_EQ(output[0].dims(2), batchsize);
}

TEST_F(ModuleTestF16, TransformerFwdF16) {
  if (!af::isHalfAvailable(af::getDevice())) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  int batchsize = 10;
  int timesteps = 120;
  int c = 32;
  int nheads = 4;

  auto tr =
      Transformer(c, c / nheads, c, nheads, timesteps, 0.2, 0.1, false, false);
  auto input =
      Variable(af::randu(c, timesteps, batchsize, 1, af::dtype::f16), false);

  auto output = tr.forward({input});
  if (OptimMode::get().getOptimLevel() == OptimLevel::O3) {
    ASSERT_EQ(output[0].type(), input.type());
  } else {
    ASSERT_EQ(output[0].type(), af::dtype::f32); // result is upcast
  }

  ASSERT_EQ(output[0].dims(0), c);
  ASSERT_EQ(output[0].dims(1), timesteps);
  ASSERT_EQ(output[0].dims(2), batchsize);
}

TEST(ModuleTest, PositionEmbeddingFwd) {
  int batchsize = 10;
  int timesteps = 120;
  int csz = 256;

  auto posemb = PositionEmbedding(csz, timesteps, 0.5);
  auto input = Variable(af::randu(csz, timesteps, batchsize, 1), false);

  auto output = posemb.forward({input});

  ASSERT_EQ(output[0].dims(0), csz);
  ASSERT_EQ(output[0].dims(1), timesteps);
  ASSERT_EQ(output[0].dims(2), batchsize);

  ASSERT_FALSE(allClose(output[0], input));
}

TEST_F(ModuleTestF16, PositionEmbeddingFwdF16) {
  if (!af::isHalfAvailable(af::getDevice())) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  int batchsize = 10;
  int timesteps = 120;
  int csz = 256;

  auto posemb = PositionEmbedding(csz, timesteps, 0.5);
  auto input =
      Variable(af::randu(csz, timesteps, batchsize, 1, af::dtype::f16), false);

  auto output = posemb.forward({input});

  ASSERT_EQ(output[0].dims(0), csz);
  ASSERT_EQ(output[0].dims(1), timesteps);
  ASSERT_EQ(output[0].dims(2), batchsize);

  ASSERT_FALSE(allClose(output[0], input));
}

TEST(ModuleTest, SinusoidalPositionEmbeddingFwd) {
  int batchsize = 10;
  int timesteps = 120;
  int csz = 256;

  auto posemb = SinusoidalPositionEmbedding(csz, 2.);
  auto input = Variable(af::randu(csz, timesteps, batchsize, 1), false) - 0.5;

  auto output = posemb.forward({input});

  ASSERT_EQ(output[0].dims(0), csz);
  ASSERT_EQ(output[0].dims(1), timesteps);
  ASSERT_EQ(output[0].dims(2), batchsize);
  ASSERT_TRUE((af::max(output[0].array())).scalar<float>() <= 2);
  ASSERT_TRUE((af::min(output[0].array())).scalar<float>() >= -2);
}

TEST_F(ModuleTestF16, SinusoidalPositionEmbeddingFwdF16) {
  if (!af::isHalfAvailable(af::getDevice())) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  int batchsize = 10;
  int timesteps = 120;
  int csz = 256;

  auto posemb = SinusoidalPositionEmbedding(csz, 2.);
  auto input =
      Variable(af::randu(csz, timesteps, batchsize, 1, af::dtype::f16), false) - 0.5;

  auto output = posemb.forward({input});

  ASSERT_EQ(output[0].dims(0), csz);
  ASSERT_EQ(output[0].dims(1), timesteps);
  ASSERT_EQ(output[0].dims(2), batchsize);
  auto outfp16 = output[0].as(af::dtype::f32).array();
  ASSERT_TRUE((af::max(outfp16)).scalar<float>() <= 2);
  ASSERT_TRUE((af::min(outfp16)).scalar<float>() >= -2);
}


TEST(ModuleTest, TDSFwd) {
  int batchsize = 10;
  int timesteps = 120;
  int w = 4;
  int c = 10;

  auto tds = TDSBlock(c, 9, w);
  auto input = Variable(af::randu(timesteps, w, c, batchsize), false);

  auto output = tds.forward({input})[0];

  ASSERT_EQ(output.dims(0), timesteps);
  ASSERT_EQ(output.dims(1), w);
  ASSERT_EQ(output.dims(2), c);
}

TEST(ModuleTest, StreamingTDSFwd) {
  int batchsize = 10;
  int timesteps = 120;
  int w = 4;
  int c = 10;
  int kw = 9;
  int rPad = 3;

  auto stds =
      TDSBlock(c, kw, w, 0 /* dropout */, 0 /* innerLinearDim */, rPad, true);

  auto input = Variable(af::randu(timesteps, w, c, batchsize), false);

  auto output = stds.forward({input})[0];

  ASSERT_EQ(output.dims(0), timesteps);
  ASSERT_EQ(output.dims(1), w);
  ASSERT_EQ(output.dims(2), c);
}

TEST(ModuleTest, SpecAugmentFwd) {
  SpecAugment specAug(0, 27, 2, 100, 0.2, 2);
  int T = 512, F = 80;
  auto input = Variable(af::randu(T, F), false);

  specAug.eval();
  ASSERT_TRUE(fl::allClose(input, specAug(input)));

  specAug.train();
  auto output = specAug(input);
  ASSERT_FALSE(fl::allClose(input, output));

  // Every value of output is either 0 or input
  for (int t = 0; t < T; ++t) {
    for (int f = 0; f < F; ++f) {
      auto o = output.array()(t, f).scalar<float>();
      auto i = input.array()(t, f).scalar<float>();
      ASSERT_TRUE(o == i || o == 0);
    }
  }

  // non-zero time frames are masked
  int tZeros = 0;
  for (int t = 0; t < T; ++t) {
    auto curOutSlice = output.array().row(t);
    tZeros = af::allTrue<bool>(curOutSlice == 0) ? tZeros + 1 : tZeros;
  }
  ASSERT_GT(tZeros, 0);

  // non-zero frequency channels are masked
  int fZeros = 0;
  for (int f = 0; f < F; ++f) {
    auto curOutSlice = output.array().col(f);
    fZeros = af::allTrue<bool>(curOutSlice == 0) ? fZeros + 1 : fZeros;
  }
  ASSERT_GT(fZeros, 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
