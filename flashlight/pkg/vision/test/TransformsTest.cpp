/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/CocoTransforms.h"

#include <gtest/gtest.h>

bool allClose(
    const af::array& a,
    const af::array& b,
    const double precision = 1e-5) {
  if ((a.numdims() != b.numdims()) || (a.dims() != b.dims())) {
    return false;
  }
  return (af::max<double>(af::abs(a - b)) < precision);
}

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

  std::vector<af::array> in = {af::constant(1.0, {256, 256, 10}),
                               af::array(),
                               af::array(),
                               af::array(),
                               af::array({4, 2}, bboxesVector.data()),
                               af::constant(0.0, {1, 2})};

  // Crop from x, y (10, 10), with target heigh and width to be ten
  std::vector<af::array> out = fl::pkg::vision::crop(in, 10, 5, 20, 25);
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
  af::array expOut = af::array({4, 2}, expVector.data());
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

  std::vector<af::array> in = {
      af::constant(1.0, {256, 256, 10}),
      af::array(),
      af::array(),
      af::array(),
      af::array({numElementsPerBoxes, numBoxes}, bboxesVector.data()),
      af::iota({1, 3})};

  // Crop from x, y (10, 10), with target heigh and width to be ten
  std::vector<af::array> out = fl::pkg::vision::crop(in, 5, 5, 100, 100);
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
  af::array expOut = af::array({4, 2}, expVector.data());
  std::vector<float> expClassVector = {0, 2};
  af::array expClassOut = af::array({1, 2}, expClassVector.data());
  ASSERT_TRUE(allClose(expOut, outBoxes, 1e-5));
  ASSERT_TRUE(allClose(expClassOut, outClasses, 1e-5));
}
