/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <string>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/fl/tensor/TensorBase.h"

using namespace fl;

TEST(SerializationTest, Residual) {
  std::shared_ptr<Residual> model = std::make_shared<Residual>();
  model->add(Linear(12, 6));
  model->add(Linear(6, 6));
  model->add(ReLU());
  model->addShortcut(1, 3);
  const fs::path path = fs::temp_directory_path() / "Residual.mdl";
  save(path, model);

  std::shared_ptr<Residual> loaded;
  load(path, loaded);

  auto input = Variable(fl::rand({12, 10, 3, 4}), false);
  auto output = model->forward(input);
  auto outputl = loaded->forward(input);

  ASSERT_TRUE(allParamsClose(*loaded.get(), *model));
  ASSERT_TRUE(allClose(outputl, output));
}

TEST(SerializationTest, AsymmetricConv1D) {
  int c = 32;
  auto model = std::make_shared<AsymmetricConv1D>(c, c, 5, 1, -1, 0, 1);

  const fs::path path = fs::temp_directory_path() / "AsymmetricConv1D.mdl";
  save(path, model);

  std::shared_ptr<AsymmetricConv1D> loaded;
  load(path, loaded);

  auto input = Variable(fl::rand({25, 10, c, 4}), false);
  auto output = model->forward(input);
  auto outputl = loaded->forward(input);

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl, output));
}

TEST(SerializationTest, Transformer) {
  int batchsize = 10;
  int timesteps = 120;
  int c = 32;
  int nheads = 4;

  auto model = std::make_shared<Transformer>(
      c, c / nheads, c, nheads, timesteps, 0.2, 0.1, false, false);
  model->eval();

  const fs::path path = fs::temp_directory_path() / "Transformer.mdl";
  save(path, model);

  std::shared_ptr<Transformer> loaded;
  load(path, loaded);
  loaded->eval();

  // auto input = Variable(fl::rand({c, timesteps, batchsize, 1}), false);
  auto input = Variable(fl::rand({c, timesteps, batchsize}), false);
  auto output = model->forward({input, Variable()});
  auto outputl = loaded->forward({input, Variable()});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

TEST(SerializationTest, ConformerSerialization) {
  int batchsize = 10;
  int timesteps = 120;
  int c = 32;
  int nheads = 4;

  auto model = std::make_shared<Conformer>(
      c, c / nheads, c, nheads, timesteps, 33, 0.2, 0.1);
  model->eval();

  const fs::path path = fs::temp_directory_path() / "Conformer.mdl";
  save(path, model);

  std::shared_ptr<Conformer> loaded;
  load(path, loaded);
  loaded->eval();

  // auto input = Variable(fl::rand({c, timesteps, batchsize, 1}), false);
  auto input = Variable(fl::rand({c, timesteps, batchsize}), false);
  auto output = model->forward({input, Variable()});
  auto outputl = loaded->forward({input, Variable()});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

TEST(SerializationTest, PositionEmbedding) {
  auto model = std::make_shared<PositionEmbedding>(128, 100, 0.1);
  model->eval();

  const fs::path path = fs::temp_directory_path() / "PositionEmbedding.mdl";
  save(path, model);

  std::shared_ptr<PositionEmbedding> loaded;
  load(path, loaded);
  loaded->eval();

  // auto input = Variable(fl::rand({128, 10, 5, 1}), false);
  auto input = Variable(fl::rand({128, 10, 5}), false);
  auto output = model->forward({input});
  auto outputl = loaded->forward({input});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

TEST(SerializationTest, SinusoidalPositionEmbedding) {
  auto model = std::make_shared<SinusoidalPositionEmbedding>(128, 2.);

  const fs::path path =
      fs::temp_directory_path() / "SinusoidalPositionEmbedding.mdl";
  save(path, model);

  std::shared_ptr<SinusoidalPositionEmbedding> loaded;
  load(path, loaded);

  // auto input = Variable(fl::rand({128, 10, 5, 1}), false);
  auto input = Variable(fl::rand({128, 10, 5}), false);
  auto output = model->forward({input});
  auto outputl = loaded->forward({input});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

TEST(SerializationTest, AdaptiveEmbedding) {
  std::vector<int> cutoff = {5, 10, 25};
  auto model = std::make_shared<AdaptiveEmbedding>(128, cutoff);

  const fs::path path = fs::temp_directory_path() / "AdaptiveEmbedding.mdl";
  save(path, model);

  std::shared_ptr<AdaptiveEmbedding> loaded;
  load(path, loaded);

  std::vector<int> values = {1, 4, 6, 2, 12, 7, 4, 21, 22, 18, 3, 23};
  auto input =
      Variable(Tensor::fromVector({6, 2}, values, fl::dtype::f32), false);
  auto output = model->forward(input);
  auto outputl = loaded->forward(input);

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl, output));
}

TEST(SerializationTest, RawWavSpecAugment) {
  auto model = std::make_shared<RawWavSpecAugment>(
      0, 1, 1, 0, 0, 0, 1, 2000, 6000, 16000, 20000);
  model->eval();

  const fs::path path = fs::temp_directory_path() / "RawWavSpecAugment.mdl";
  save(path, model);

  std::shared_ptr<RawWavSpecAugment> loaded;
  load(path, loaded);
  loaded->train();

  int T = 300;
  // Input is T x C x B (here, C, B = 1)
  auto time = 2 * M_PI * fl::reshape(fl::iota({T}), {T, 1, 1}) / 16000;
  auto finalWav = fl::sin(time * 500) + fl::sin(time * 1000) +
      fl::sin(time * 7000) + fl::sin(time * 7500);
  auto inputWav = finalWav + fl::sin(time * 3000) + fl::sin(time * 4000) +
      fl::sin(time * 5000);

  auto filteredWav = loaded->forward(fl::Variable(inputWav, false));
  // compare middle of filtered wave to avoid edge artifacts comparison
  int halfKernelWidth = 63;
  ASSERT_TRUE(fl::allClose(
      fl::Variable(
          finalWav(fl::range(halfKernelWidth, T - halfKernelWidth)), false),
      filteredWav(fl::range(halfKernelWidth, T - halfKernelWidth)),
      1e-3));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
