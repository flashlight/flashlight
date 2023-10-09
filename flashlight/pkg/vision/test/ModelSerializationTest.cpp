/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/pkg/vision/models/ViT.h"
#include "flashlight/pkg/vision/nn/VisionTransformer.h"

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;
using namespace fl::pkg::vision;

TEST(SerializationTest, VisionTransformer) {
  int hiddenEmbSize = 768;
  int nHeads = 12;
  int mlpSize = 3072;

  auto model = std::make_shared<VisionTransformer>(
      hiddenEmbSize, hiddenEmbSize / nHeads, mlpSize, nHeads, 0, 0);
  model->eval();

  const fs::path path = fs::temp_directory_path() / "VisionTransformer.mdl";
  save(path, model);

  std::shared_ptr<VisionTransformer> loaded;
  load(path, loaded);
  loaded->eval();

  auto input = Variable(fl::rand({hiddenEmbSize, 197, 20}), false);
  auto output = model->forward({input});
  auto outputl = loaded->forward({input});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

TEST(SerializationTest, ViT) {
  int hiddenEmbSize = 768;
  int nHeads = 12;
  int mlpSize = 3072;

  auto model = std::make_shared<fl::pkg::vision::ViT>(
      12, // FLAGS_model_layers,
      hiddenEmbSize,
      mlpSize,
      nHeads,
      0.1, // setting non-zero drop prob for testing purpose
      0.1, // setting non-zero drop prob for testing purpose
      1000);
  model->eval();

  const fs::path path = fs::temp_directory_path() / "ViT.mdl";
  save(path, model);

  std::shared_ptr<ViT> loaded;
  load(path, loaded);
  loaded->eval();

  auto input = Variable(fl::rand({224, 224, 3, 20}), false);
  auto output = model->forward({input});
  auto outputl = loaded->forward({input});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
