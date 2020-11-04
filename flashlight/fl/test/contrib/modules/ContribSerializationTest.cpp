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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
