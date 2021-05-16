/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <arrayfire.h>

#include "flashlight/pkg/vision/dataset/BoxUtils.h"

#include <gtest/gtest.h>

TEST(BoxUtils, IOU) {
  std::vector<float> labels = {0, 0, 10, 10, 1};
  std::vector<float> preds = {1, 1, 11, 11, 1};
  af::array labelArr = af::array(5, 1, labels.data());
  af::array predArr = af::array(5, 1, preds.data());
  af_print(box_iou(labelArr, predArr));
}

TEST(BoxUtils, IOU2) {
  std::vector<float> labels = {0, 0, 10, 12};
  std::vector<float> preds = {12, 12, 22, 22};
  af::array labelArr = af::array(5, 1, labels.data());
  af::array predArr = af::array(5, 1, preds.data());
  af_print(box_iou(labelArr, predArr));
}
