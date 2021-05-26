/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>

#include "flashlight/pkg/vision/dataset/BoxUtils.h"

#include <gtest/gtest.h>

using namespace fl::pkg::vision;

TEST(BoxUtils, IOU) {
  std::vector<float> labels = {0, 0, 10, 10, 1};
  std::vector<float> preds = {1, 1, 11, 11, 1};
  std::vector<float> costs = {0.680672268908};
  fl::Variable labelArr = {af::array(5, 1, labels.data()), false};
  fl::Variable predArr = {af::array(5, 1, preds.data()), false};
  fl::Variable result, uni;
  std::tie(result, uni) = boxIou(predArr, labelArr);
  EXPECT_EQ(result.array()(0, 0).scalar<float>(), costs[0]);
}

TEST(BoxUtils, IOU2) {
  std::vector<float> labels = {0, 0, 10, 10, 1};
  std::vector<float> preds = {12, 12, 22, 22, 1};
  std::vector<float> costs = {0.0};
  fl::Variable labelArr = {af::array(5, 1, labels.data()), false};
  fl::Variable predArr = {af::array(5, 1, preds.data()), false};
  fl::Variable result, uni;
  std::tie(result, uni) = boxIou(predArr, labelArr);
  EXPECT_EQ(result(0, 0).array().scalar<float>(), costs[0]);
}

TEST(BoxUtils, IOU3) {
  std::vector<float> labels = {0, 0, 2, 2, 1};
  std::vector<float> preds = {1, 1, 3, 3, 1};
  std::vector<float> costs = {0.142857142857};
  fl::Variable labelArr = fl::Variable(af::array(5, 1, labels.data()), false);
  fl::Variable predArr = fl::Variable(af::array(5, 1, preds.data()), false);
  fl::Variable result, uni;
  std::tie(result, uni) = boxIou(predArr, labelArr);
  EXPECT_EQ(result(0, 0).array().scalar<float>(), costs[0]);
}

TEST(BoxUtils, IOU4) {
  std::vector<float> labels = {0, 0, 2, 2, 1};
  std::vector<float> preds = {3, 0, 5, 2, 1};
  std::vector<float> costs = {0.0};
  fl::Variable labelArr = {af::array(5, 1, labels.data()), false};
  fl::Variable predArr = {af::array(5, 1, preds.data()), false};
  fl::Variable result, uni;
  std::tie(result, uni) = boxIou(predArr, labelArr);
  EXPECT_EQ(result(0, 0).array().scalar<float>(), costs[0]);
}

TEST(BoxUtils, IOU5) {
  std::vector<float> labels = {0, 0, 2, 2, 1};
  std::vector<float> preds = {1, 1, 3, 3, 1};
  std::vector<float> costs = {0.14285714285714285};
  fl::Variable labelArr = {af::array(5, 1, labels.data()), false};
  fl::Variable predArr = {af::array(5, 1, preds.data()), false};
  fl::Variable result, uni;
  std::tie(result, uni) = boxIou(predArr, labelArr);
  EXPECT_EQ(result.array()(0, 0).scalar<float>(), costs[0]);
}

TEST(BoxUtils, IOU6) {
  std::vector<float> labels = {0, 0, 4, 4, 1};
  std::vector<float> preds = {1, 1, 3, 3, 1};
  std::vector<float> costs = {0.25};
  fl::Variable labelArr = {af::array(5, 1, labels.data()), false};
  fl::Variable predArr = {af::array(5, 1, preds.data()), false};
  fl::Variable result, uni;
  std::tie(result, uni) = boxIou(predArr, labelArr);
  EXPECT_EQ(result(0, 0).array().scalar<float>(), costs[0]);
}

TEST(BoxUtils, IOU7) {
  std::vector<float> preds = {1, 1, 3, 3, 1, 0, 1, 2, 3, 1};
  std::vector<float> labels = {0, 0, 4, 4, 1, 0, 0, 2, 2, 1};
  std::vector<float> costs = {
      0.25,
      0.25, // Both boxes are contained in first box
      0.14285714285714285,
      0.3333333333 //
  };
  fl::Variable labelArr = {af::array(5, 2, labels.data()), false};
  fl::Variable predArr = {af::array(5, 2, preds.data()), false};
  fl::Variable result, uni;
  std::tie(result, uni) = boxIou(predArr, labelArr);
  EXPECT_EQ(result(0, 0).array().scalar<float>(), costs[0]);
  EXPECT_EQ(result(1, 0).array().scalar<float>(), costs[1]);
  EXPECT_EQ(result(0, 1).array().scalar<float>(), costs[2]);
  EXPECT_EQ(result(1, 1).array().scalar<float>(), costs[3]);
}

TEST(BoxUtils, IOU8) {
  std::vector<float> preds = {1, 1, 3, 3, 1, 0, 1, 2, 3, 1};
  std::vector<float> labels = {
      0,
      0,
      4,
      4,
      1,
  };
  std::vector<float> costs = {
      0.25, 0.25, // Both boxes are contained in first box
  };
  fl::Variable labelArr = {af::array(5, 2, labels.data()), false};
  fl::Variable predArr = {af::array(5, 2, preds.data()), false};
  fl::Variable result, uni;
  std::tie(result, uni) = boxIou(predArr, labelArr);
  EXPECT_EQ(result(0, 0).array().scalar<float>(), costs[0]);
  EXPECT_EQ(result(1, 0).array().scalar<float>(), costs[1]);
}

TEST(BoxUtils, IOUBatched) {
  std::vector<float> preds = {1, 1, 3, 3, 1, 0, 1, 2, 3, 1};
  std::vector<float> labels = {
      0,
      0,
      4,
      4,
      1,
      0,
      0,
      4,
      4,
      1,
  };
  std::vector<float> costs = {
      0.25, 0.25, // Both boxes are contained in first box
  };
  fl::Variable labelArr = {af::array(5, 1, 2, labels.data()), false};
  fl::Variable predArr = {af::array(5, 1, 2, preds.data()), false};
  fl::Variable result, uni;
  std::tie(result, uni) = boxIou(predArr, labelArr);
  EXPECT_EQ(result(0, 0, 0).array().scalar<float>(), costs[0]);
  EXPECT_EQ(result(0, 0, 1).array().scalar<float>(), costs[1]);
}
//// Test GIOU
//// The first box is further away from the second box and should have a smaller
//// score
TEST(BoxUtils, GIOU) {
  std::vector<float> preds = {0, 0, 1, 1, 1, 1, 1, 2, 2, 1};
  std::vector<float> labels = {2, 2, 3, 3, 1};
  fl::Variable labelArr = {af::array(5, 1, labels.data()), false};
  fl::Variable predArr = {af::array(5, 2, preds.data()), false};
  fl::Variable result = generalizedBoxIou(predArr, labelArr);
  EXPECT_LT(result(0, 0).array().scalar<float>(), result(1, 0).scalar<float>());
}
