/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <string>

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/nn/nn.h"

using namespace fl;

namespace {

std::string getTmpPath(const std::string& key) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  return std::string("/tmp/test_") + userstr + key + std::string(".mdl");
}

} // namespace

TEST(ModuleTest, ResidualSerialization) {
  std::shared_ptr<Residual> model = std::make_shared<Residual>();
  model->add(Linear(12, 6));
  model->add(Linear(6, 6));
  model->add(ReLU());
  model->addShortcut(1, 3);
  save(getTmpPath("Residual"), model);

  std::shared_ptr<Residual> loaded;
  load(getTmpPath("Residual"), loaded);

  auto input = Variable(af::randu(12, 10, 3, 4), false);
  auto output = model->forward(input);
  auto outputl = loaded->forward(input);

  ASSERT_TRUE(allParamsClose(*loaded.get(), *model));
  ASSERT_TRUE(allClose(outputl, output));
}

TEST(ModuleTest, AsymmetricConv1DSerialization) {
  int c = 32;
  auto model = std::make_shared<AsymmetricConv1D>(c, c, 5, 1, -1, 0, 1);

  save(getTmpPath("AsymmetricConv1D"), model);

  std::shared_ptr<AsymmetricConv1D> loaded;
  load(getTmpPath("AsymmetricConv1D"), loaded);

  auto input = Variable(af::randu(25, 10, c, 4), false);
  auto output = model->forward(input);
  auto outputl = loaded->forward(input);

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl, output));
}

TEST(ModuleTest, TransformerSerialization) {
  int batchsize = 10;
  int timesteps = 120;
  int c = 32;
  int nheads = 4;

  auto model = std::make_shared<Transformer>(
    c, c / nheads, c, nheads, timesteps, 0.2, 0.1, false, false);
  model->eval();
  save(getTmpPath("Transformer"), model);

  std::shared_ptr<Transformer> loaded;
  load(getTmpPath("Transformer"), loaded);
  loaded->eval();

  auto input = Variable(af::randu(c, timesteps, batchsize, 1), false);
  auto output = model->forward({input});
  auto outputl = loaded->forward({input});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

TEST(ModuleTest, ConformerSerialization) {
  int batchsize = 10;
  int timesteps = 120;
  int c = 32;
  int nheads = 4;

  auto model = std::make_shared<Conformer>(
    c, c / nheads, c, nheads, timesteps, 33, 0.2, 0.1);
  model->eval();
  save(getTmpPath("Conformer"), model);

  std::shared_ptr<Conformer> loaded;
  load(getTmpPath("Conformer"), loaded);
  loaded->eval();

  auto input = Variable(af::randu(c, timesteps, batchsize, 1), false);
  auto output = model->forward({input});
  auto outputl = loaded->forward({input});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

TEST(ModuleTest, PositionEmbedding) {
  auto model = std::make_shared<PositionEmbedding>(128, 100, 0.1);
  model->eval();

  save(getTmpPath("PositionEmbedding"), model);

  std::shared_ptr<PositionEmbedding> loaded;
  load(getTmpPath("PositionEmbedding"), loaded);
  loaded->eval();

  auto input = Variable(af::randu(128, 10, 5, 1), false);
  auto output = model->forward({input});
  auto outputl = loaded->forward({input});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

TEST(ModuleTest, SinusoidalPositionEmbedding) {
  auto model = std::make_shared<SinusoidalPositionEmbedding>(128, 2.);

  save(getTmpPath("SinusoidalPositionEmbedding"), model);

  std::shared_ptr<SinusoidalPositionEmbedding> loaded;
  load(getTmpPath("SinusoidalPositionEmbedding"), loaded);

  auto input = Variable(af::randu(128, 10, 5, 1), false);
  auto output = model->forward({input});
  auto outputl = loaded->forward({input});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

TEST(ModuleTest, AdaptiveEmbedding) {
  std::vector<int> cutoff = {5, 10, 25};
  auto model = std::make_shared<AdaptiveEmbedding>(128, cutoff);

  save(getTmpPath("AdaptiveEmbedding"), model);

  std::shared_ptr<AdaptiveEmbedding> loaded;
  load(getTmpPath("AdaptiveEmbedding"), loaded);

  std::vector<int> values = {1, 4, 6, 2, 12, 7, 4, 21, 22, 18, 3, 23};
  auto input = Variable(af::array(af::dim4(6, 2), values.data()), false);
  auto output = model->forward(input);
  auto outputl = loaded->forward(input);

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl, output));
}

TEST(ModuleTest, RawWavSpecAugment) {
  auto model = std::make_shared<RawWavSpecAugment>(0, 1, 1, 0, 0, 0, 1, 2000, 6000, 16000, 20000);
  model->eval();

  save(getTmpPath("RawWavSpecAugment"), model);

  std::shared_ptr<RawWavSpecAugment> loaded;
  load(getTmpPath("RawWavSpecAugment"), loaded);
  loaded->train();

  int T = 300;
  auto time = 2 * M_PI * af::iota(af::dim4(T)) / 16000;
  auto finalWav =
    af::sin(time * 500) + af::sin(time * 1000) + af::sin(time * 7000) + af::sin(time * 7500);
  auto inputWav = finalWav + af::sin(time * 3000) + af::sin(time * 4000) + af::sin(time * 5000);

  auto filteredWav = loaded->forward(fl::Variable(inputWav, false));
  // compare middle of filtered wave to avoid edge artifacts comparison
  int halfKernelWidth = 63;
  ASSERT_TRUE(fl::allClose(
    fl::Variable(finalWav.rows(halfKernelWidth, T - halfKernelWidth - 1), false),
    filteredWav.rows(halfKernelWidth, T - halfKernelWidth - 1),
    1e-3));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
