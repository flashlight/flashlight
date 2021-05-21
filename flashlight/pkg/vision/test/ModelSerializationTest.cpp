/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/pkg/vision/models/ViT.h"
#include "flashlight/pkg/vision/nn/VisionTransformer.h"

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

using namespace fl;
using namespace fl::ext::image;

TEST(SerializationTest, VisionTransformer) {
  int hiddenEmbSize = 768;
  int nHeads = 12;
  int mlpSize = 3072;

  auto model = std::make_shared<VisionTransformer>(
      hiddenEmbSize, hiddenEmbSize / nHeads, mlpSize, nHeads, 0, 0);
  model->eval();

  const std::string path = fl::lib::getTmpPath("VisionTransformer.mdl");
  save(path, model);

  std::shared_ptr<VisionTransformer> loaded;
  load(path, loaded);
  loaded->eval();

  auto input = Variable(af::randu(hiddenEmbSize, 197, 20, 1), false);
  auto output = model->forward({input});
  auto outputl = loaded->forward({input});

  ASSERT_TRUE(allParamsClose(*loaded, *model));
  ASSERT_TRUE(allClose(outputl[0], output[0]));
}

TEST(SerializationTest, ViT) {
  int hiddenEmbSize = 768;
  int nHeads = 12;
  int mlpSize = 3072;

  auto model = std::make_shared<fl::ext::image::ViT>(
      12, // FLAGS_model_layers,
      hiddenEmbSize,
      mlpSize,
      nHeads,
      0.1, // setting non-zero drop prob for testing purpose
      0.1, // setting non-zero drop prob for testing purpose
      1000);
  model->eval();

  const std::string path = fl::lib::getTmpPath("ViT.mdl");
  save(path, model);

  std::shared_ptr<ViT> loaded;
  load(path, loaded);
  loaded->eval();

  auto input = Variable(af::randu(224, 224, 3, 20), false);
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
