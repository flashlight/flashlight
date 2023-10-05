/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/pkg/vision/dataset/CocoTransforms.h"

#include <gtest/gtest.h>

using namespace fl;

TEST(Crop, CropBasic) {
  std::vector<float> bboxesVector = {
      10,
      10,
      20,
      20, // box1
      20,
      20,
      30,
      30 // box2
  };

  std::vector<Tensor> in = {
      fl::full({256, 256, 10}, 1.),
      Tensor(),
      Tensor(),
      Tensor(),
      Tensor::fromVector({4, 2}, bboxesVector),
      fl::full({1, 2}, 0.)};

  // Crop from x, y (10, 10), with target heigh and width to be ten
  std::vector<Tensor> out = fl::pkg::vision::crop(in, 10, 5, 20, 25);
  auto outBoxes = out[4];
  std::vector<float> expVector = {
      0,
      5,
      10,
      15, // box1
      10,
      15,
      20,
      25 // box2
  };
  Tensor expOut = Tensor::fromVector({4, 2}, expVector);
  ASSERT_TRUE(allClose(expOut, outBoxes, 1e-5));
}

TEST(Crop, CropClip) {
  int numBoxes = 3;
  int numElementsPerBoxes = 4;

  std::vector<float> bboxesVector = {
      0,
      0,
      100,
      100, // box1
      0,
      0,
      4,
      4, // box3 // will be removed
      5,
      5,
      105,
      105 // box2
  };

  std::vector<Tensor> in = {
      fl::full({256, 256, 10}, 1.),
      Tensor(),
      Tensor(),
      Tensor(),
      Tensor::fromVector({numElementsPerBoxes, numBoxes}, bboxesVector),
      fl::iota({1, 3})};

  // Crop from x, y (10, 10), with target heigh and width to be ten
  std::vector<Tensor> out = fl::pkg::vision::crop(in, 5, 5, 100, 100);
  auto outBoxes = out[4];
  auto outClasses = out[5];
  std::vector<float> expVector = {
      0,
      0,
      95,
      95, // box1
      0,
      0,
      100,
      100 // box2
  };
  Tensor expOut = Tensor::fromVector({4, 2}, expVector);
  std::vector<float> expClassVector = {0, 2};
  Tensor expClassOut = Tensor::fromVector({1, 2}, expClassVector);
  ASSERT_TRUE(allClose(expOut, outBoxes, 1e-5));
  ASSERT_TRUE(allClose(expClassOut, outClasses, 1e-5));
}
