/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

namespace {

class ContainerTestClass : public Sequential {
 public:
  ContainerTestClass() = default;
  ContainerTestClass(const ContainerTestClass& other) {
    copy(other);
  }
  ContainerTestClass& operator=(const ContainerTestClass& other) {
    copy(other);
    return *this;
  }
  ContainerTestClass(ContainerTestClass&& other) = default;
  ContainerTestClass& operator=(ContainerTestClass&& other) = default;
  void copy(const ContainerTestClass& other) {
    auto orphanParamIdxMap = other.getOrphanedParamsIdxMap();
    for (int i = -1; i < static_cast<int>(other.modules_.size()); ++i) {
      if (i >= 0) {
        add(other.modules_[i]->clone());
      }
      auto [paramIter, pEnd] = orphanParamIdxMap.equal_range(i);
      for (; paramIter != pEnd; ++paramIter) {
        const auto& param = other.params_[paramIter->second];
        params_.emplace_back(param.copy());
      }
    }
  }

  std::unique_ptr<Module> clone() const override {
    return std::make_unique<ContainerTestClass>(*this);
  }

  void addParam(const Variable& param) {
    params_.push_back(param);
  }
};

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

TEST(ModuleTest, EmbeddingFwd) {
  int embDim = 3, nEmb = 5, nQuery = 2, batchSize = 2;
  auto wtVar = param(Tensor::fromVector<float>(
      {embDim, nEmb}, {8, 2, 2, 10, 5, 3, 3, 4, 6, 12, 3, 8, 0, 5, 2}));

  auto inVar = input(Tensor::fromVector<float>({2, batchSize}, {1, 3, 0, 0}));

  auto expectedOutVar = Variable(
      Tensor::fromVector<float>(
          {embDim, nQuery, batchSize}, {10, 5, 3, 12, 3, 8, 8, 2, 2, 8, 2, 2}),
      true);

  // Var initialization
  auto emb = Embedding(wtVar);
  ASSERT_TRUE(allClose(emb.forward(inVar), expectedOutVar, 1E-7));

  // Regular initialization
  emb = Embedding(embDim, nEmb);
  wtVar = emb.param(0);
  ASSERT_EQ(wtVar.shape(), Shape({embDim, nEmb}));

  expectedOutVar = Variable(
      fl::reshape(
          wtVar.tensor()(fl::span, inVar.tensor()),
          {embDim, nQuery, batchSize}),
      true);
  ASSERT_TRUE(allClose(emb.forward(inVar), expectedOutVar, 1E-7));
}

TEST(ModuleTest, LinearFwd) {
  int n_in = 2, n_out = 3, x = 4, batchsize = 2;
  auto wtVar =
      param(Tensor::fromVector<float>({n_out, n_in}, {8, 2, 2, 10, 5, 3}));

  auto inVar = input(Tensor::fromVector<float>(
      {n_in, x, batchsize}, {6, 2, 1, 4, 8, 2, 7, 1, 10, 7, 3, 7, 5, 9, 2, 4}));

  auto expected_outVar = Variable(
      Tensor::fromVector<float>(
          {n_out, x, batchsize},
          {68,  22, 18, 48, 22, 14, 84,  26, 22, 66, 19, 17,
           150, 55, 41, 94, 41, 27, 130, 55, 37, 56, 24, 16}),
      true);

  auto linNoBias = Linear(wtVar);
  ASSERT_TRUE(allClose(linNoBias.forward(inVar), expected_outVar, 1E-7));

  auto bsVar = input(Tensor::fromVector<float>({n_out}, {1, 2, 3}));
  expected_outVar = Variable(
      Tensor::fromVector<float>(
          {n_out, x, batchsize},
          {69,  24, 21, 49, 24, 17, 85,  28, 25, 67, 21, 20,
           151, 57, 44, 95, 43, 30, 131, 57, 40, 57, 26, 19}),
      true);

  auto linBias = Linear(wtVar, bsVar);
  ASSERT_TRUE(allClose(linBias.forward(inVar), expected_outVar, 1E-7));
}

TEST_F(ModuleTestF16, LinearFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  int n_in = 2, n_out = 3, x = 4, batchsize = 2;
  auto wtVar =
      param(Tensor::fromVector<float>({n_out, n_in}, {8, 2, 2, 10, 5, 3}));

  auto inVar = input(Tensor::fromVector<float>(
                         {n_in, x, batchsize},
                         {6, 2, 1, 4, 8, 2, 7, 1, 10, 7, 3, 7, 5, 9, 2, 4})
                         .astype(fl::dtype::f16));

  auto expected_outVar = Variable(
      Tensor::fromVector<float>(
          {n_out, x, batchsize},
          {68,  22, 18, 48, 22, 14, 84,  26, 22, 66, 19, 17,
           150, 55, 41, 94, 41, 27, 130, 55, 37, 56, 24, 16})
          .astype(fl::dtype::f16),
      true);

  auto linNoBias = Linear(wtVar);
  auto result = linNoBias.forward(inVar);
  ASSERT_EQ(result.type(), inVar.type());
  ASSERT_TRUE(allClose(result, expected_outVar, 1E-2));

  auto bsVar = input(Tensor::fromVector<float>({n_out}, {1, 2, 3}));
  ;
  expected_outVar = Variable(
      Tensor::fromVector<float>(
          {n_out, x, batchsize},
          {69,  24, 21, 49, 24, 17, 85,  28, 25, 67, 21, 20,
           151, 57, 44, 95, 43, 30, 131, 57, 40, 57, 26, 19})
          .astype(inVar.type()),
      true);

  auto linBias = Linear(wtVar, bsVar);
  auto resultBias = linBias.forward(inVar);
  ASSERT_EQ(resultBias.type(), fl::dtype::f16);
  ASSERT_TRUE(allClose(resultBias, expected_outVar, 1E-3));

  // OptimLevel::O3 is active with this fixture
  ASSERT_EQ(linBias.forward(inVar.astype(fl::dtype::f32)).type(), fl::dtype::f16);
}

TEST(ModuleTest, ConvPadding) {
  auto conv1 = Conv2D(30, 100, 3, 5, 2, 1, PaddingMode::SAME, 0, true, 1);
  auto conv2 = Conv2D(
      30, 100, 3, 5, 2, 1, PaddingMode::SAME, PaddingMode::SAME, true, 1);
  auto conv3 =
      Conv2D(30, 100, 10, 10, 1, 1, PaddingMode::SAME, PaddingMode::SAME, 4, 4);
  auto input = Variable(fl::rand({32, 32, 30, 2}), false);

  auto conv1Op = conv1(input);
  ASSERT_EQ(conv1Op.shape(), Shape({16, 28, 100, 2}));

  auto conv2Op = conv2(input);
  ASSERT_EQ(conv2Op.shape(), Shape({16, 32, 100, 2}));

  // test dilation
  auto conv3Op = conv3(input);
  ASSERT_EQ(conv3Op.shape(), Shape({32, 32, 100, 2}));
}

TEST(ModuleTest, GLUFwd) {
  auto inVar = Variable(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.2, 0.2, 0.1, 0.5, 0.3}), true);

  auto expected_outVar = Variable(
      Tensor::fromVector<float>({3, 1}, {0.419983, 0.124492, 0.114888}), true);

  GatedLinearUnit glu(1);
  ASSERT_TRUE(allClose(glu.forward(inVar), expected_outVar, 1E-4));

  // test batching
  int batchsize = 5;
  inVar = Variable(fl::rand({10, 7, batchsize}), true);
  glu = GatedLinearUnit(0);

  auto batchOutVar = glu(inVar);

  for (int i = 0; i < batchsize; ++i) {
    expected_outVar = glu.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }
}

TEST_F(ModuleTestF16, GLUFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto inVar = Variable(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.2, 0.2, 0.1, 0.5, 0.3})
          .astype(fl::dtype::f16),
      true);

  auto expected_outVar = Variable(
      Tensor::fromVector<float>({3, 1}, {0.419983, 0.124492, 0.114888})
          .astype(fl::dtype::f16),
      true);

  GatedLinearUnit glu(1);
  auto out = glu.forward(inVar);
  ASSERT_EQ(out.type(), inVar.type());
  ASSERT_TRUE(allClose(out, expected_outVar, 1E-2));

  // test batching
  int batchsize = 5;
  inVar = Variable(fl::rand({10, 7, batchsize}).astype(fl::dtype::f16), true);
  glu = GatedLinearUnit(0);

  auto batchOutVar = glu(inVar);

  for (int i = 0; i < batchsize; ++i) {
    expected_outVar = glu.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_EQ(batchOutVar.type(), expected_outVar.type());
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-3));
  }
}

TEST(ModuleTest, LogSoftmaxFwd) {
  auto inVar = Variable(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.2, 0.2, 0.1, 0.5, 0.3}), true);

  auto expected_outVar0 = Variable(
      Tensor::fromVector<float>(
          {3, 2}, {-0.740805, -1.34081, -1.34081, -1.3119, -0.911902, -1.1119}),
      true);
  LogSoftmax lsm0(0);
  ASSERT_TRUE(allClose(lsm0.forward(inVar), expected_outVar0, 1E-4));

  auto expected_outVar1 = Variable(
      Tensor::fromVector<float>(
          {3, 2},
          {-0.403186, -0.854355, -0.744397, -1.10319, -0.554355, -0.644397}),
      true);
  LogSoftmax lsm1(1);
  ASSERT_TRUE(allClose(lsm1.forward(inVar), expected_outVar1, 1E-4));

  // test batching
  int batchsize = 5;
  inVar = Variable(fl::rand({10, 7, batchsize}), true);
  LogSoftmax lsm(0);

  auto batchOutVar = lsm(inVar);

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar =
        lsm.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }
}

TEST_F(ModuleTestF16, LogSoftmaxFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto inVar = Variable(
      Tensor::fromVector<float>({3, 2}, {0.8, 0.2, 0.2, 0.1, 0.5, 0.3})
          .astype(fl::dtype::f16),
      true);

  auto expected_outVar0 = Variable(
      Tensor::fromVector<float>(
          {3, 2}, {-0.740805, -1.34081, -1.34081, -1.3119, -0.911902, -1.1119}),
      true);
  LogSoftmax lsm0(0);
  auto result0 = lsm0.forward(inVar);
  ASSERT_TRUE(allClose(result0, expected_outVar0, 1E-3));

  auto expected_outVar1 = Variable(
      Tensor::fromVector<float>(
          {3, 2},
          {-0.403186, -0.854355, -0.744397, -1.10319, -0.554355, -0.644397}),
      true);
  LogSoftmax lsm1(1);
  ASSERT_TRUE(allClose(lsm1.forward(inVar), expected_outVar1, 1E-3));

  // test batching
  int batchsize = 5;
  inVar = Variable(fl::rand({10, 7, batchsize}), true);
  LogSoftmax lsm(0);

  auto batchOutVar = lsm(inVar);

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar =
        lsm.forward(inVar(fl::span, fl::span, fl::range(i, i + 1)));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }
}

TEST(ModuleTest, ConvolutionFwd) {
  // test batching
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2, 1, 1, true, 1);
  int batchsize = 10;
  auto input = fl::rand({120, 100, 30, batchsize}, fl::dtype::f32);
  auto batchOutVar = conv(Variable(input, false));

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar = conv(
        Variable(input(fl::span, fl::span, fl::span, fl::range(i, i + 1)), false));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-5));
  }
}

TEST_F(ModuleTestF16, ConvolutionFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // test batching
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2, 1, 1, true, 1);
  int batchsize = 1;
  auto input = fl::rand({120, 100, 30, batchsize}, fl::dtype::f16);
  auto batchOutVar = conv(Variable(input, false));
  ASSERT_EQ(batchOutVar.type(), input.type());

  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar = conv(
        Variable(input(fl::span, fl::span, fl::span, fl::range(i, i + 1)), false));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }

  auto inputF32 = fl::rand({120, 100, 30, batchsize}, fl::dtype::f32);
  ASSERT_EQ(
      conv(Variable(input, false)).type(),
      fl::dtype::f16); // OptimLevel::O3 is active with this fixture
}

TEST(ModuleTest, ConvolutionWithGroupFwd) {
  // test batching
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2, true, 2);
  int batchsize = 10;
  auto input = fl::rand({120, 100, 30, batchsize});
  auto batchOutVar = conv(Variable(input, false));
  for (int i = 0; i < batchsize; ++i) {
    auto expected_outVar = conv(
        Variable(input(fl::span, fl::span, fl::span, fl::range(i, i + 1)), false));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-5));
  }
}

TEST(ModuleTest, PoolingFwd) {
  // test batching
  auto pool = Pool2D(9, 7, 1, 1, PaddingMode::SAME, PaddingMode::SAME);
  int batchsize = 10;
  auto input = fl::rand({120, 100, 30, batchsize});
  auto batchOutVar = pool(Variable(input, false));
  for (int i = 0; i < batchsize; ++i) {
    ASSERT_EQ(input.shape(), batchOutVar.shape());
    auto expected_outVar = pool(
        Variable(input(fl::span, fl::span, fl::span, fl::range(i, i + 1)), false));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
        1E-7));
  }
}

TEST_F(ModuleTestF16, PoolingFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  // test batching
  auto pool = Pool2D(9, 7, 1, 1, PaddingMode::SAME, PaddingMode::SAME);
  int batchsize = 10;
  auto input = fl::rand({120, 100, 30, batchsize}, fl::dtype::f16);
  auto batchOutVar = pool(Variable(input, false));
  for (int i = 0; i < batchsize; ++i) {
    ASSERT_EQ(input.shape(), batchOutVar.shape());
    auto expected_outVar = pool(
        Variable(input(fl::span, fl::span, fl::span, fl::range(i, i + 1)), false));
    ASSERT_TRUE(allClose(
        batchOutVar.tensor()(fl::span, fl::span, fl::span, fl::range(i, i + 1)),
        expected_outVar.tensor(),
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
      fl::rand({input_size, batch_size, seq_length}, fl::dtype::f32), true);
  unsigned n_params = 51;
  auto w = Variable(fl::rand({1, 1, n_params}, fl::dtype::f32), true);
  for (int i = 0; i < in.elements(); ++i) {
    in.tensor().flat(i) = (i + 1) * 0.01;
  }
  for (int i = 0; i < w.elements(); ++i) {
    w.tensor().flat(i) = (i + 1) * 0.01;
  }
  auto rnn = RNN(input_size, hidden_size, num_layers, mode);
  rnn.setParams(w, 0);

  auto out = rnn(in);
  Shape expected_dims({3, 5, 6});
  ASSERT_EQ(out.shape(), expected_dims);
  // Calculated from Lua Torch Cudnn implementation

  auto expected_outVar = Variable(
      Tensor::fromVector<float>(
          expected_dims,
          {1.5418,  1.6389,  1.7361,  1.5491,  1.6472,  1.7452,  1.5564,
           1.6554,  1.7544,  1.5637,  1.6637,  1.7636,  1.5710,  1.6719,
           1.7728,  3.4571,  3.7458,  4.0345,  3.4761,  3.7670,  4.0578,
           3.4951,  3.7881,  4.0812,  3.5141,  3.8093,  4.1045,  3.5331,
           3.8305,  4.1278,  5.6947,  6.2004,  6.7060,  5.7281,  6.2373,
           6.7466,  5.7614,  6.2743,  6.7871,  5.7948,  6.3112,  6.8276,
           5.8282,  6.3482,  6.8681,  8.2005,  8.9458,  9.6911,  8.2500,
           9.0005,  9.7509,  8.2995,  9.0551,  9.8107,  8.3491,  9.1098,
           9.8705,  8.3986,  9.1645,  9.9303,  10.9520, 11.9587, 12.9655,
           11.0191, 12.0326, 13.0462, 11.0861, 12.1065, 13.1269, 11.1532,
           12.1804, 13.2075, 11.2203, 12.2543, 13.2882, 13.9432, 15.2333,
           16.5233, 14.0291, 15.3277, 16.6263, 14.1149, 15.4221, 16.7292,
           14.2008, 15.5165, 16.8322, 14.2866, 15.6109, 16.9351}),
      true);
  ASSERT_TRUE(allClose(out, expected_outVar, 1E-4));
}

TEST(ModuleTest, LSTMFwd) {
  auto mode = RnnMode::LSTM;
  int num_layers = 4;
  int hidden_size = 5;
  int input_size = 3;
  int batch_size = 2;
  int seq_length = 2;

  auto in = Variable(
      fl::rand({input_size, batch_size, seq_length}, fl::dtype::f32), true);
  unsigned n_params = 920;
  auto w = Variable(fl::rand({1, 1, n_params}, fl::dtype::f32), true);

  for (int i = 0; i < in.elements(); ++i) {
    in.tensor().flat(i) = (i + 1) * 0.001;
  }
  for (int i = 0; i < w.elements(); ++i) {
    w.tensor().flat(i) = (i + 1) * 0.001;
  }

  auto rnn = RNN(input_size, hidden_size, num_layers, mode);
  rnn.setParams(w, 0);

  auto out = rnn(in);
  Shape expected_dims({5, 2, 2});
  ASSERT_EQ(out.shape(), expected_dims);
  // Calculated from Lua Torch Cudnn implementation
  auto expected_outVar = Variable(
      Tensor::fromVector<float>(
          expected_dims,
          {0.7390, 0.7395, 0.7399, 0.7403, 0.7407, 0.7390, 0.7395,
           0.7399, 0.7403, 0.7407, 0.9617, 0.9618, 0.9619, 0.9619,
           0.962,  0.9617, 0.9618, 0.9619, 0.9619, 0.962}),
      true);
  ASSERT_TRUE(allClose(out, expected_outVar, 1E-4));
}

TEST(ModuleTest, GRUFwd) {
  auto mode = RnnMode::GRU;
  int num_layers = 4;
  int hidden_size = 5;
  int input_size = 3;
  int batch_size = 2;
  int seq_length = 2;

  auto in = Variable(
      fl::rand({input_size, batch_size, seq_length}, fl::dtype::f32), true);
  unsigned n_params = 690;
  auto w = Variable(fl::rand({1, 1, n_params}, fl::dtype::f32), true);

  for (int i = 0; i < in.elements(); ++i) {
    in.tensor().flat(i) = (i + 1) * 0.001;
  }
  for (int i = 0; i < w.elements(); ++i) {
    w.tensor().flat(i) = (i + 1) * 0.001;
  }

  auto rnn = RNN(input_size, hidden_size, num_layers, mode);
  rnn.setParams(w, 0);

  auto out = rnn(in);
  Shape expected_dims({5, 2, 2});
  ASSERT_EQ(out.shape(), expected_dims);
  // Calculated from Lua Torch Cudnn implementation
  auto expected_outVar = Variable(
      Tensor::fromVector<float>(
          expected_dims,
          {0.1430, 0.1425, 0.1419, 0.1413, 0.1408, 0.1430, 0.1425,
           0.1419, 0.1413, 0.1408, 0.2206, 0.2194, 0.2181, 0.2168,
           0.2155, 0.2206, 0.2194, 0.2181, 0.2168, 0.2155}),
      true);
  ASSERT_TRUE(allClose(out, expected_outVar, 1E-4));
}

TEST_F(ModuleTestF16, RNNFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto mode = RnnMode::RELU;
  int num_layers = 2;
  int hidden_size = 3;
  int input_size = 4;
  int batch_size = 5;
  int seq_length = 6;

  auto in = Variable(
      fl::rand({input_size, batch_size, seq_length}, fl::dtype::f16), true);
  unsigned n_params = 51;
  auto w = Variable(fl::rand({1, 1, n_params}, fl::dtype::f16), true);
  for (int i = 0; i < in.elements(); ++i) {
    in.tensor().flat(i) = (i + 1) * 0.01;
  }
  for (int i = 0; i < w.elements(); ++i) {
    w.tensor().flat(i) = (i + 1) * 0.01;
  }
  auto rnn = RNN(input_size, hidden_size, num_layers, mode);
  rnn.setParams(w, 0);

  auto out = rnn(in);
  Shape expected_dims({3, 5, 6});
  ASSERT_EQ(out.shape(), expected_dims);
  // Calculated from Lua Torch Cudnn implementation
  auto expected_outVar = Variable(
      Tensor::fromVector<float>(
          expected_dims,
          {1.5418,  1.6389,  1.7361,  1.5491,  1.6472,  1.7452,  1.5564,
           1.6554,  1.7544,  1.5637,  1.6637,  1.7636,  1.5710,  1.6719,
           1.7728,  3.4571,  3.7458,  4.0345,  3.4761,  3.7670,  4.0578,
           3.4951,  3.7881,  4.0812,  3.5141,  3.8093,  4.1045,  3.5331,
           3.8305,  4.1278,  5.6947,  6.2004,  6.7060,  5.7281,  6.2373,
           6.7466,  5.7614,  6.2743,  6.7871,  5.7948,  6.3112,  6.8276,
           5.8282,  6.3482,  6.8681,  8.2005,  8.9458,  9.6911,  8.2500,
           9.0005,  9.7509,  8.2995,  9.0551,  9.8107,  8.3491,  9.1098,
           9.8705,  8.3986,  9.1645,  9.9303,  10.9520, 11.9587, 12.9655,
           11.0191, 12.0326, 13.0462, 11.0861, 12.1065, 13.1269, 11.1532,
           12.1804, 13.2075, 11.2203, 12.2543, 13.2882, 13.9432, 15.2333,
           16.5233, 14.0291, 15.3277, 16.6263, 14.1149, 15.4221, 16.7292,
           14.2008, 15.5165, 16.8322, 14.2866, 15.6109, 16.9351}),
      true);
  ASSERT_TRUE(allClose(out, expected_outVar.astype(in.type()), 5E-2));
}

TEST(ModuleTest, ViewFwd) {
  auto module = View(Shape({-1, 0, 6}));
  auto input = Variable(Tensor({1, 2, 3, 4}), true);
  ASSERT_EQ(module(input).shape(), Shape({2, 2, 6}));
}

TEST(ModuleTest, DropoutFwd) {
  auto module = Dropout(0.5);
  // Train Mode
  module.train();
  auto in = Variable(fl::rand({1000, 1000}), true);
  auto out = module(in);

  ASSERT_NEAR(
      out.elements() - fl::countNonzero(out.tensor()).scalar<unsigned>(),
      in.elements() / 2,
      in.elements() / 16); // Check enough zeroes

  ASSERT_GT(
      fl::amax(out.tensor()).scalar<float>(), 1.5); // Check input is scaled

  // Eval Mode
  module.eval();
  out = module(in);
  ASSERT_TRUE(allClose(out, in, 1E-5));
}

TEST_F(ModuleTestF16, DropoutFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  auto module = Dropout(0.5);
  // Train Mode
  module.train();
  auto in = Variable(fl::rand({1000, 1000}, fl::dtype::f16), true);
  auto out = module(in);
  ASSERT_EQ(out.type(), fl::dtype::f16);

  ASSERT_NEAR(
      out.elements() - fl::countNonzero(out.tensor()).scalar<unsigned>(),
      in.elements() / 2,
      in.elements() / 16); // Check enough zeroes

  ASSERT_GT(
      fl::amax(out.tensor()).asScalar<float>(), 1.5); // Check input is scaled

  // Eval Mode
  module.eval();
  out = module(in);
  ASSERT_TRUE(allClose(out, in, 1E-5));
}

TEST(ModuleTest, PaddingFwd) {
  auto module = Padding({{1, 2}, {3, 4}}, -1);
  auto input = Variable(fl::rand({1, 2, 3, 4}, fl::dtype::f64), true);
  auto output = module(input);
  ASSERT_EQ(output.shape(), Shape({4, 9, 3, 4}));
  ASSERT_TRUE(allClose(input, output(fl::range(1, 2), fl::range(3, 5))));
  ASSERT_NEAR(
      fl::sum(input.tensor()).scalar<double>(),
      fl::sum(output.tensor()).scalar<double>() + 408,
      1E-5);
}

TEST(ModuleTest, LayerNormFwd) {
  double eps = 1E-5;
  std::vector<int> feat_axes = {3};
  int F = 10;
  auto input = Variable(fl::rand({4, 4, 3, F}), true);

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

  ASSERT_TRUE(allClose(out.tensor(), true_out.tensor(), eps));
  ASSERT_EQ(out.type(), input.type());

  // with affine transform
  auto module2 = LayerNorm(feat_axes, eps, true);

  module2.train();
  auto out_train = module2.forward(input);
  module2.eval();
  auto out_eval = module2.forward(input);

  ASSERT_TRUE(allClose(out_train.tensor(), out_eval.tensor(), eps));
  ASSERT_EQ(out_train.shape(), input.shape());

  // with affine transform
  auto module3 = LayerNorm(feat_axes, eps, true, F);
  module3.setParams(Variable(fl::full({F}, 1.0), false), 0);
  module3.setParams(Variable(fl::full({F}, 0.0), false), 1);
  auto out3 = module3.forward(input);
  ASSERT_TRUE(allClose(out_train.tensor(), out3.tensor(), eps));

  // With other shapes
  auto input3Dim = Variable(fl::rand({4, 4, 3}), true);
  auto module4 = LayerNorm(std::vector<int>{0}, eps, false);
  out = module4.forward(input3Dim);
  ASSERT_EQ(out.shape(), input3Dim.shape());
}

TEST_F(ModuleTestF16, LayerNormFwdF16) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half-precision not supported on this device";
  }

  double eps = 5E-2;
  std::vector<int> feat_axes = {3};
  auto input = Variable(fl::rand({4, 4, 3, 10}, fl::dtype::f16), true);

  auto sample_mean = mean(input, {3});
  auto sample_var = var(input, {3}, true);
  auto true_out = (input - tileAs(sample_mean, input).astype(input.type())) /
      tileAs(fl::sqrt(sample_var + eps), input).astype(input.type());

  // no affine transform
  auto module1 = LayerNorm(feat_axes, eps, false);

  module1.train();
  auto out = module1.forward(input);

  ASSERT_TRUE(allClose(out, true_out.astype(out.type()), eps));

  module1.eval();
  out = module1.forward(input);

  ASSERT_TRUE(
      allClose(out.tensor(), true_out.tensor().astype(out.type()), eps));

  // with affine transform
  auto module2 = LayerNorm(feat_axes, eps, true);

  module2.train();
  auto out_train = module2.forward(input);
  module2.eval();
  auto out_eval = module2.forward(input);

  ASSERT_TRUE(allClose(out_train.tensor(), out_eval.tensor(), eps));
  ASSERT_EQ(out_train.shape(), input.shape());

  module2.train();
}

TEST(ModuleTest, NormalizeFwd) {
  auto input = Variable(fl::rand({10, 3}, fl::dtype::f64), true);
  auto module = Normalize({1}, 2, 1e-12, 5);
  module.train();
  auto out = module.forward(input);
  ASSERT_TRUE(allClose(
      fl::sqrt(fl::sum(out.tensor() * out.tensor(), {1})),
      fl::full({10}, 5, fl::dtype::f64)));
}

TEST(ModuleTest, TransformFwd) {
  auto inVar = Variable(fl::full({4, 5}, 1.0), true);

  auto l = Transform([](const Variable& in) { return fl::log(in); });

  ASSERT_TRUE(allClose(l.forward(inVar).tensor(), fl::full(inVar.shape(), 0.0)));
}

TEST(ModuleTest, PrecisionCastFwd) {
  if (!fl::f16Supported()) {
    GTEST_SKIP() << "Half precision not available on this device";
  }

  auto in = Variable(fl::full({3, 3}, 1.0), true);
  auto precisionCast = PrecisionCast(fl::dtype::f16);
  auto out = precisionCast.forward(in);

  ASSERT_EQ(out.type(), fl::dtype::f16);
  ASSERT_TRUE(allClose(in.tensor(), out.astype(fl::dtype::f32).tensor()));
}

TEST(ModuleTest, ContainerReplaceParam) {
  auto seq = ContainerTestClass();
  seq.addParam(Variable(fl::rand({5, 5}), true));
  seq.add(Linear(10, 20));
  seq.addParam(Variable(fl::rand({5, 5}), true));
  seq.add(ReLU());
  seq.add(Linear(20, 30));
  seq.addParam(Variable(fl::rand({5, 5}), true));

  // Change the first parameter
  auto new_param = Variable(fl::rand({5, 5}), true);
  seq.setParams(new_param, 0);
  ASSERT_TRUE(allClose(seq.params()[0], new_param));

  // Change the first linear layer's first parameter
  new_param = Variable(fl::rand({10, 20}), true);
  seq.setParams(new_param, 1);
  ASSERT_TRUE(allClose(seq.params()[1], new_param));
  ASSERT_TRUE(allClose(seq.module(0)->param(0), new_param));

  // Change the second linear layer's first parameter
  new_param = Variable(fl::rand({20, 30}), true);
  seq.setParams(new_param, 4);
  ASSERT_TRUE(allClose(seq.params()[4], new_param));
  ASSERT_TRUE(allClose(seq.module(2)->param(0), new_param));

  // Change the last parameter
  new_param = Variable(fl::rand({5, 5}), true);
  seq.setParams(new_param, 6);
  ASSERT_TRUE(allClose(seq.param(6), new_param));
}

TEST(ModuleTest, AdaptiveSoftMaxPredict) {
  // test predict gives the same as argmax along probs
  int N = 5;
  int C = 5;
  int T = 10;
  int B = 5;

  auto x = input(fl::rand({N, T, B}, fl::dtype::f32));
  auto y = Variable(
      (fl::rand({T, B}, fl::dtype::u32) % C).astype(fl::dtype::s32), false);

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);

  auto probs = activation->forward(x);
  auto result1 = activation->predict(x).tensor();
  auto result2 = fl::argmax(probs.tensor(), 0, /* keepDims = */ true);

  ASSERT_TRUE(allClose(result1, result2));
}

TEST(ModuleTest, AdaptiveSoftMaxLossBatchFwd) {
  // test batching
  int N = 5;
  int C = 3;
  int T = 10;
  int B = 5;

  auto x = input(fl::rand({N, T, B}, fl::dtype::f32));
  auto y = Variable(
      (fl::rand({T, B}, fl::dtype::u32) % C).astype(fl::dtype::s32), false);

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);
  auto asml =
      std::make_shared<AdaptiveSoftMaxLoss>(activation, ReduceMode::NONE);
  auto batchOutVar = asml->forward(x, y);

  auto singleOut = fl::full({T, B}, 0, fl::dtype::f32);
  for (int i = 0; i < B; ++i) {
    auto singleOutVar = asml->forward(
        x(fl::span, fl::span, fl::range(i, i + 1)), y(fl::span, fl::range(i, i + 1)));
    singleOut(fl::span, i) = singleOutVar.tensor();
  }

  ASSERT_TRUE(allClose(batchOutVar.tensor(), singleOut));
}

TEST(ModuleTest, AdaptiveSoftMaxLossIgnoreIndex) {
  // test batching
  int N = 5;
  int C = 3;
  int T = 10;
  int B = 5;

  auto x = input(fl::rand({N, T, B}, fl::dtype::f32));
  auto y = Variable(
      (fl::rand({T, B}, fl::dtype::u32) % C).astype(fl::dtype::s32), false);
  auto ignoreIdx = y(0, 0).scalar<int>();
  auto ignoreCount = fl::sum(y.tensor() != ignoreIdx).scalar<unsigned>();

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);
  auto asml1 = std::make_shared<AdaptiveSoftMaxLoss>(
      activation, ReduceMode::SUM, ignoreIdx);
  auto asml2 = std::make_shared<AdaptiveSoftMaxLoss>(
      activation, ReduceMode::MEAN, ignoreIdx);

  auto lossSum = asml1->forward(x, y);
  auto lossMean = asml2->forward(x, y);
  ASSERT_NEAR(
      fl::sum(lossSum.tensor()).scalar<float>(),
      fl::sum(lossMean.tensor()).scalar<float>() * ignoreCount,
      1E-5);
}

TEST(ModuleTest, IdentityFwd) {
  auto module = Identity();
  std::vector<Variable> in = {
      Variable(fl::rand({1000, 1000}), true),
      Variable(fl::rand({100, 100}), true)};

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

TEST(ModuleTest, ModuleCloneCopy) {
  int n_in = 1, n_out = 2;
  auto wtVar = param(Tensor::fromVector<float>({n_out, n_in}, {2, 4}));
  auto inVar = input(Tensor::fromVector<float>({n_in}, {3}));
  Variable expected_outVar(Tensor::fromVector<float>({n_out}, {6, 12}), true);

  Linear lin(wtVar);
  ASSERT_TRUE(allClose(lin(inVar), expected_outVar, 1E-7));

  // Intentionally cast to base Module ptr and clone/copy via the various
  // options
  std::unique_ptr<Module> modulePtr = std::make_unique<Linear>(std::move(lin));
  std::unique_ptr<Module> clonedModulePtr = modulePtr->clone();

  // Change the original module param and check the cloned modules have not
  // changed
  modulePtr->param(0).tensor() += 1.0F;
  ASSERT_FALSE(
      allClose(modulePtr->forward({inVar}).front(), expected_outVar, 1E-7));

  ASSERT_TRUE(allClose(
      clonedModulePtr->forward({inVar}).front(), expected_outVar, 1E-7));
}

TEST(ModuleTest, ContainerCloneCopy) {
  ContainerTestClass seq;
  seq.addParam(Variable(fl::rand({5, 5}), true));
  seq.add(Linear(10, 20));
  // Create copy/clone vis copy constructor
  auto seqCopy = seq;

  // Make sure they are the same
  ASSERT_TRUE(allClose(seq.params()[0], seqCopy.params()[0]));
  ASSERT_TRUE(allClose(seq.params()[1], seqCopy.params()[1]));

  // Change the first parameter and check the copy has not changed
  Variable new_param(fl::rand({5, 5}), true);
  seq.setParams(new_param, 0);
  ASSERT_TRUE(allClose(seq.params()[0], new_param));
  ASSERT_FALSE(allClose(seqCopy.params()[0], seq.params()[0]));

  // Change the linear layer's first parameter and check the copy has not
  // changed
  new_param = Variable(fl::rand({10, 20}), true);
  seq.setParams(new_param, 1);
  ASSERT_TRUE(allClose(seq.params()[1], new_param));
  ASSERT_TRUE(allClose(seq.module(0)->param(0), new_param));
  ASSERT_FALSE(allClose(seqCopy.params()[1], seq.params()[1]));
  ASSERT_FALSE(allClose(seqCopy.module(0)->param(0), seq.module(0)->param(0)));

  // Intentionally cast to base Module ptr and clone/copy via the various
  // options
  std::unique_ptr<Module> modulePtr =
      std::make_unique<ContainerTestClass>(std::move(seq));
  std::unique_ptr<Module> clonedModulePtr = modulePtr->clone();

  ASSERT_TRUE(allClose(clonedModulePtr->params()[0], modulePtr->params()[0]));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
