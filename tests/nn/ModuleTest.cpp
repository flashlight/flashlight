/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <gtest/gtest.h>

#include "flashlight/autograd/autograd.h"
#include "flashlight/common/common.h"
#include "flashlight/nn/nn.h"

using namespace fl;

namespace {

class ContainerTestClass : public Sequential {
 public:
  void addParam(const Variable& param) {
    params_.push_back(param);
  }
};

} // namespace

TEST(ModuleTest, EmbeddingFwd) {
  int embDim = 3, nEmb = 5, nQuery = 2, batchSize = 2;
  std::array<float, 15> wt = {8, 2, 2, 10, 5, 3, 3, 4, 6, 12, 3, 8, 0, 5, 2};
  auto wtVar = param(af::array(embDim, nEmb, wt.data()));

  std::array<float, 4> in = {1, 3, 0, 0};
  auto inVar = input(af::array(2, batchSize, in.data()));

  std::array<float, 12> expectedOut = {10, 5, 3, 12, 3, 8, 8, 2, 2, 8, 2, 2};
  auto expectedOutVar =
      Variable(af::array(embDim, nQuery, batchSize, expectedOut.data()), true);

  // Var initialization
  auto emb = Embedding(wtVar);
  ASSERT_TRUE(allClose(emb.forward(inVar), expectedOutVar, 1E-7));

  // Regular initialization
  emb = Embedding(embDim, nEmb);
  wtVar = emb.param(0);
  ASSERT_EQ(wtVar.dims(), af::dim4(embDim, nEmb));
  expectedOutVar = Variable(
      af::moddims(
          af::lookup(wtVar.array(), af::flat(inVar.array()), 1),
          af::dim4(embDim, nQuery, batchSize)),
      true);
  ASSERT_TRUE(allClose(emb.forward(inVar), expectedOutVar, 1E-7));
}

TEST(ModuleTest, LinearFwd) {
  int n_in = 2, n_out = 3, x = 4, batchsize = 2;
  std::array<float, 6> wt = {8, 2, 2, 10, 5, 3};
  auto wtVar = param(af::array(n_out, n_in, wt.data()));

  std::array<float, 16> in = {6, 2, 1, 4, 8, 2, 7, 1, 10, 7, 3, 7, 5, 9, 2, 4};
  auto inVar = input(af::array(n_in, x, batchsize, in.data()));

  std::array<float, 24> expected_out = {68, 22, 18,  48, 22,  14, 84, 26,
                                        22, 66, 19,  17, 150, 55, 41, 94,
                                        41, 27, 130, 55, 37,  56, 24, 16};
  auto expected_outVar =
      Variable(af::array(n_out, x, batchsize, expected_out.data()), true);

  auto linNoBias = Linear(wtVar);
  ASSERT_TRUE(allClose(linNoBias.forward(inVar), expected_outVar, 1E-7));

  std::array<float, 3> bs = {1, 2, 3};
  auto bsVar = input(af::array(n_out, bs.data()));
  expected_out = {69,  24, 21, 49, 24, 17, 85,  28, 25, 67, 21, 20,
                  151, 57, 44, 95, 43, 30, 131, 57, 40, 57, 26, 19};
  expected_outVar =
      Variable(af::array(n_out, x, batchsize, expected_out.data()), true);

  auto linBias = Linear(wtVar, bsVar);
  ASSERT_TRUE(allClose(linBias.forward(inVar), expected_outVar, 1E-7));
}

TEST(ModuleTest, ConvPadding) {
  auto conv1 = Conv2D(30, 100, 3, 5, 2, 1, PaddingMode::SAME, 0, true, 1);
  auto conv2 = Conv2D(
      30, 100, 3, 5, 2, 1, PaddingMode::SAME, PaddingMode::SAME, true, 1);
  auto conv3 =
      Conv2D(30, 100, 10, 10, 1, 1, PaddingMode::SAME, PaddingMode::SAME, 4, 4);
  auto input = Variable(af::randu(32, 32, 30, 2), false);

  auto conv1Op = conv1(input);
  ASSERT_EQ(conv1Op.dims(), af::dim4(16, 28, 100, 2));

  auto conv2Op = conv2(input);
  ASSERT_EQ(conv2Op.dims(), af::dim4(16, 32, 100, 2));

  // test dilation
  auto conv3Op = conv3(input);
  ASSERT_EQ(conv3Op.dims(), af::dim4(32, 32, 100, 2));
}

TEST(ModuleTest, GLUFwd) {
  std::array<float, 6> in = {0.8, 0.2, 0.2, 0.1, 0.5, 0.3};
  auto inVar = Variable(af::array(3, 2, in.data()), true);

  std::array<float, 3> expected_out = {0.419983, 0.124492, 0.114888};
  auto expected_outVar = Variable(af::array(3, expected_out.data()), true);

  GatedLinearUnit glu(1);
  ASSERT_TRUE(allClose(glu.forward(inVar), expected_outVar, 1E-4));

  // test batching
  int batchsize = 5;
  inVar = Variable(af::randu(10, 7, batchsize), true);
  glu = GatedLinearUnit(0);

  auto batchOutVar = glu(inVar);

  for (int i = 0; i < batchsize; ++i) {
    expected_outVar = glu.forward(inVar.slice(i));
    ASSERT_TRUE(allClose(
        batchOutVar.array()(af::span, af::span, i),
        expected_outVar.array(),
        1E-7));
  }
}

TEST(ModuleTest, LogSoftmaxFwd) {
  std::array<float, 6> in = {0.8, 0.2, 0.2, 0.1, 0.5, 0.3};
  auto inVar = Variable(af::array(3, 2, in.data()), true);

  std::array<float, 6> expected_out0 = {
      -0.740805, -1.34081, -1.34081, -1.3119, -0.911902, -1.1119};
  auto expected_outVar0 = Variable(af::array(3, 2, expected_out0.data()), true);
  LogSoftmax lsm0(0);
  ASSERT_TRUE(allClose(lsm0.forward(inVar), expected_outVar0, 1E-4));

  std::array<float, 6> expected_out1 = {
      -0.403186, -0.854355, -0.744397, -1.10319, -0.554355, -0.644397};
  auto expected_outVar1 = Variable(af::array(3, 2, expected_out1.data()), true);
  LogSoftmax lsm1(1);
  ASSERT_TRUE(allClose(lsm1.forward(inVar), expected_outVar1, 1E-4));

  // test batching
  int batchsize = 5;
  inVar = Variable(af::randu(10, 7, batchsize), true);
  LogSoftmax lsm(0);

  auto batchOutVar = lsm(inVar);

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar = lsm.forward(inVar.slice(i));
    ASSERT_TRUE(allClose(
        batchOutVar.array()(af::span, af::span, i),
        expected_outVar.array(),
        1E-7));
  }
}

TEST(ModuleTest, ConvolutionFwd) {
  // test batching
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2, true);
  int batchsize = 10;
  auto input = af::randu(120, 100, 30, batchsize);
  auto batchOutVar = conv(Variable(input, false));
  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar =
        conv(Variable(input(af::span, af::span, af::span, i), false));
    ASSERT_TRUE(allClose(
        batchOutVar.array()(af::span, af::span, af::span, i),
        expected_outVar.array(),
        1E-7));
  }
}

TEST(ModuleTest, ConvolutionWithGroupFwd) {
  // test batching
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2, true, 2);
  int batchsize = 10;
  auto input = af::randu(120, 100, 30, batchsize);
  auto batchOutVar = conv(Variable(input, false));
  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar =
        conv(Variable(input(af::span, af::span, af::span, i), false));
    ASSERT_TRUE(allClose(
        batchOutVar.array()(af::span, af::span, af::span, i),
        expected_outVar.array(),
        1E-7));
  }
}

TEST(ModuleTest, PoolingFwd) {
  // test batching
  auto pool = Pool2D(9, 7, 1, 1, PaddingMode::SAME, PaddingMode::SAME);
  int batchsize = 10;
  auto input = af::randu(120, 100, 30, batchsize);
  auto batchOutVar = pool(Variable(input, false));
  for (int i = 0; i < batchsize; ++i) {
    ASSERT_EQ(input.dims(), batchOutVar.dims());
    auto expected_outVar =
        pool(Variable(input(af::span, af::span, af::span, i), false));
    ASSERT_TRUE(allClose(
        batchOutVar.array()(af::span, af::span, af::span, i),
        expected_outVar.array(),
        1E-7));
  }
}

TEST(ModuleTest, RNNFwd) {
  auto mode = RnnMode::RELU;
  int num_layers = 2;
  int hidden_size = 3;
  int input_size = 4;
  int batch_size = 5;
  int seq_length = 6;

  auto in = Variable(
      af::randu(input_size, batch_size, seq_length, af::dtype::f64), true);
  size_t n_params = 51;
  auto w = Variable(af::randu(1, 1, n_params, af::dtype::f64), true);
  for (int i = 0; i < in.elements(); ++i) {
    in.array()(i) = (i + 1) * 0.01;
  }
  for (int i = 0; i < w.elements(); ++i) {
    w.array()(i) = (i + 1) * 0.01;
  }
  auto rnn = RNN(input_size, hidden_size, num_layers, mode);
  rnn.setParams(w, 0);

  auto out = rnn(in);
  af::dim4 expected_dims(3, 5, 6);
  ASSERT_EQ(out.dims(), expected_dims);
  // Calculated from Lua Torch Cudnn implementation
  std::array<double, 90> expected_out = {
      1.5418,  1.6389,  1.7361,  1.5491,  1.6472,  1.7452,  1.5564,  1.6554,
      1.7544,  1.5637,  1.6637,  1.7636,  1.5710,  1.6719,  1.7728,  3.4571,
      3.7458,  4.0345,  3.4761,  3.7670,  4.0578,  3.4951,  3.7881,  4.0812,
      3.5141,  3.8093,  4.1045,  3.5331,  3.8305,  4.1278,  5.6947,  6.2004,
      6.7060,  5.7281,  6.2373,  6.7466,  5.7614,  6.2743,  6.7871,  5.7948,
      6.3112,  6.8276,  5.8282,  6.3482,  6.8681,  8.2005,  8.9458,  9.6911,
      8.2500,  9.0005,  9.7509,  8.2995,  9.0551,  9.8107,  8.3491,  9.1098,
      9.8705,  8.3986,  9.1645,  9.9303,  10.9520, 11.9587, 12.9655, 11.0191,
      12.0326, 13.0462, 11.0861, 12.1065, 13.1269, 11.1532, 12.1804, 13.2075,
      11.2203, 12.2543, 13.2882, 13.9432, 15.2333, 16.5233, 14.0291, 15.3277,
      16.6263, 14.1149, 15.4221, 16.7292, 14.2008, 15.5165, 16.8322, 14.2866,
      15.6109, 16.9351};

  auto expected_outVar =
      Variable(af::array(expected_dims, expected_out.data()), true);
  ASSERT_TRUE(allClose(out, expected_outVar, 1E-4));
}

TEST(ModuleTest, ViewFwd) {
  auto module = View(af::dim4(-1, 0, 6));
  auto input = Variable(af::array(1, 2, 3, 4), true);
  ASSERT_EQ(module(input).dims(), af::dim4(2, 2, 6));
}

TEST(ModuleTest, DropoutFwd) {
  auto module = Dropout(0.5);
  // Train Mode
  module.train();
  auto in = Variable(af::randu(af::dim4(1000, 1000)), true);
  auto out = module(in);

  ASSERT_NEAR(
      af::count<int>(out.array() == 0.0),
      in.elements() / 2,
      in.elements() / 16); // Check enough zeroes

  ASSERT_GT(af::max<float>(out.array()), 1.5); // Check input is scaled

  // Eval Mode
  module.eval();
  out = module(in);
  ASSERT_TRUE(allClose(out, in, 1E-5));
}

TEST(ModuleTest, PaddingFwd) {
  auto module = Padding({1, 2}, {3, 4}, -1);
  auto input = Variable(af::randu(1, 2, 3, 4, af::dtype::f64), true);
  auto output = module(input);
  ASSERT_EQ(output.dims(), af::dim4(4, 9, 3, 4));
  ASSERT_TRUE(allClose(input, output(af::seq(1, 1), af::seq(3, 4))));
  ASSERT_NEAR(
      af::sum<double>(input.array()),
      af::sum<double>(output.array()) + 408,
      1E-5);
}

TEST(ModuleTest, LayerNormFwd) {
  double eps = 1E-5;
  std::vector<int> feat_axes = {3};
  auto input = Variable(af::randu(4, 4, 3, 10), true);

  auto sample_mean = mean(input, {3});
  auto sample_var = var(input, {3}, true);
  auto true_out = (input - tileAs(sample_mean, input)) /
      tileAs(fl::sqrt(sample_var + eps), input);

  // no affine transform
  auto module1 = LayerNorm(feat_axes, eps, false);

  module1.train();
  auto out = module1.forward(input);

  ASSERT_TRUE(allClose(out, true_out, eps));
  ASSERT_EQ(out.type(), input.type());

  module1.eval();
  out = module1.forward(input);

  ASSERT_TRUE(allClose(out.array(), true_out.array(), eps));
  ASSERT_EQ(out.type(), input.type());

  // with affine transform
  auto module2 = LayerNorm(feat_axes, eps, true);

  module2.train();
  auto out_train = module2.forward(input);
  module2.eval();
  auto out_eval = module2.forward(input);

  ASSERT_TRUE(allClose(out_train.array(), out_eval.array(), eps));
  ASSERT_EQ(out_train.dims(), input.dims());
}

TEST(ModuleTest, NormalizeFwd) {
  auto input = Variable(af::randu(10, 3, af::dtype::f64), true);
  auto module = Normalize({1}, 2, 1e-12, 5);
  module.train();
  auto out = module.forward(input);
  ASSERT_TRUE(allClose(
      af::sqrt(af::sum(out.array() * out.array(), 1)),
      af::constant(5, 10, af::dtype::f64)));
}

TEST(ModuleTest, TransformFwd) {
  auto inVar = Variable(af::constant(1.0, 4, 5), true);

  auto l = Transform([](const Variable& in) { return fl::log(in); });

  ASSERT_TRUE(
      allClose(l.forward(inVar).array(), af::constant(0.0, inVar.dims())));
}

TEST(ModuleTest, ContainerReplaceParam) {
  auto seq = ContainerTestClass();
  seq.addParam(Variable(af::randu(5, 5), true));
  seq.add(Linear(10, 20));
  seq.addParam(Variable(af::randu(5, 5), true));
  seq.add(ReLU());
  seq.add(Linear(20, 30));
  seq.addParam(Variable(af::randu(5, 5), true));

  // Change the first parameter
  auto new_param = Variable(af::randu(5, 5), true);
  seq.setParams(new_param, 0);
  ASSERT_TRUE(allClose(seq.params()[0], new_param));

  // Change the first linear layer's first parameter
  new_param = Variable(af::randu(10, 20), true);
  seq.setParams(new_param, 1);
  ASSERT_TRUE(allClose(seq.params()[1], new_param));
  ASSERT_TRUE(allClose(seq.module(0)->param(0), new_param));

  // Change the second linear layer's first parameter
  new_param = Variable(af::randu(20, 30), true);
  seq.setParams(new_param, 4);
  ASSERT_TRUE(allClose(seq.params()[4], new_param));
  ASSERT_TRUE(allClose(seq.module(2)->param(0), new_param));

  // Change the last parameter
  new_param = Variable(af::randu(5, 5), true);
  seq.setParams(new_param, 6);
  ASSERT_TRUE(allClose(seq.param(6), new_param));
}

TEST(ModuleTest, AdaptiveSoftMaxPredict) {
  // test predict gives the same as argmax along probs
  int N = 5;
  int C = 5;
  int T = 10;
  int B = 5;

  auto x = input(af::randu(N, T, B, af::dtype::f32));
  auto y = Variable((af::randu(T, B, af::dtype::u32) % C).as(s32), false);

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);

  auto probs = activation->forward(x);
  auto result1 = activation->predict(x).array();
  af::array tmpValue, result2;
  af::max(tmpValue, result2, probs.array(), 0);

  ASSERT_TRUE(allClose(result1, result2));
}

TEST(ModuleTest, AdaptiveSoftMaxLossBatchFwd) {
  // test batching
  int N = 5;
  int C = 3;
  int T = 10;
  int B = 5;

  auto x = input(af::randu(N, T, B, af::dtype::f32));
  auto y = Variable((af::randu(T, B, af::dtype::u32) % C).as(s32), false);

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);
  auto asml =
      std::make_shared<AdaptiveSoftMaxLoss>(activation, ReduceMode::NONE);
  auto batchOutVar = asml->forward(x, y);

  auto singleOut = af::constant(0, T, B);
  for (int i = 0; i < B; ++i) {
    auto singleOutVar = asml->forward(x(af::span, af::span, i), y(af::span, i));
    singleOut(af::span, i) = singleOutVar.array();
  }

  ASSERT_TRUE(allClose(batchOutVar.array(), singleOut));
}

TEST(ModuleTest, AdaptiveSoftMaxLossIgnoreIndex) {
  // test batching
  int N = 5;
  int C = 3;
  int T = 10;
  int B = 5;

  auto x = input(af::randu(N, T, B, af::dtype::f32));
  auto y = Variable((af::randu(T, B, af::dtype::u32) % C).as(s32), false);
  auto ignoreIdx = y(0, 0).scalar<int>();
  auto ignoreCount = af::sum<int>(y.array() != ignoreIdx);

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);
  auto asml1 = std::make_shared<AdaptiveSoftMaxLoss>(
      activation, ReduceMode::SUM, ignoreIdx);
  auto asml2 = std::make_shared<AdaptiveSoftMaxLoss>(
      activation, ReduceMode::MEAN, ignoreIdx);

  auto lossSum = asml1->forward(x, y);
  auto lossMean = asml2->forward(x, y);
  ASSERT_NEAR(
      af::sum<float>(lossSum.array()),
      af::sum<float>(lossMean.array()) * ignoreCount,
      1E-5);
}

TEST(ModuleTest, IdentityFwd) {
  auto module = Identity();
  std::vector<Variable> in = {Variable(af::randu(af::dim4(1000, 1000)), true),
                              Variable(af::randu(af::dim4(100, 100)), true)};

  // Train Mode
  module.train();
  auto out = module(in);
  ASSERT_EQ(out.size(), 2);
  ASSERT_TRUE(allClose(out.at(0), in.at(0), 1e-20));
  ASSERT_TRUE(allClose(out.at(1), in.at(1), 1e-20));

  // Eval Mode
  module.eval();
  out = module(in);
  ASSERT_EQ(out.size(), 2);
  ASSERT_TRUE(allClose(out.at(0), in.at(0), 1e-20));
  ASSERT_TRUE(allClose(out.at(1), in.at(1), 1e-20));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
