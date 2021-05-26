/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/pkg/runtime/amp/DynamicScaler.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Utils.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/lib/common/System.h"

TEST(DynamicScalerTest, Scaling) {
  auto dynamicScaler = fl::pkg::runtime::DynamicScaler(
      32, // initFactor
      32, // maxFactor
      100 // updateInterval
  );

  auto loss = fl::uniform(af::dim4(5, 5, 5, 5));

  auto scaledLoss = dynamicScaler.scale(loss);
  ASSERT_TRUE(allClose(loss * 32, scaledLoss));

  scaledLoss.addGrad(scaledLoss);
  std::vector<fl::Variable> params{scaledLoss};
  bool unscaleStatus = dynamicScaler.unscale(params);
  ASSERT_TRUE(unscaleStatus);
  ASSERT_TRUE(allClose(loss, scaledLoss.grad()));
}

TEST(DynamicScalerTest, Serialization) {
  auto dynamicScaler = std::make_shared<fl::pkg::runtime::DynamicScaler>(
      32, // initFactor
      32, // maxFactor
      100 // updateInterval
  );

  const std::string path = fl::lib::getTmpPath("DynamicScaler.bin");
  fl::save(path, dynamicScaler);

  std::shared_ptr<fl::pkg::runtime::DynamicScaler> dynamicScaler1;
  fl::load(path, dynamicScaler1);
  ASSERT_TRUE(
      dynamicScaler->getScaleFactor() == dynamicScaler1->getScaleFactor());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
