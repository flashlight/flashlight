/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
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
