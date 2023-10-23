/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <unordered_map>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/pkg/vision/criterion/SetCriterion.h"

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::pkg::vision;

std::unordered_map<std::string, float> getLossWeights() {
  const std::unordered_map<std::string, float> lossWeightsBase = {
      {"lossCe", 1.f}, {"lossGiou", 1.f}, {"lossBbox", 1.f}};

  std::unordered_map<std::string, float> lossWeights;
  for (int i = 0; i < 6; i++) {
    for (const auto& l : lossWeightsBase) {
      std::string key = l.first + "_" + std::to_string(i);
      lossWeights[key] = l.second;
    }
  }
  return lossWeights;
}

TEST(SetCriterion, PytorchRepro) {
  const int numClasses = 80;
  const int numTargets = 1;
  const int numPreds = 1;
  const int numBatches = 1;
  std::vector<float> predBoxesVec = {2, 2, 3, 3};

  std::vector<float> targetBoxesVec = {2, 2, 3, 3};

  std::vector<float> targetClassVec = {1};
  auto predBoxes = fl::Variable(
      Tensor::fromVector({4, numPreds, numBatches, 1}, predBoxesVec), true);
  auto predLogits =
      fl::Variable(fl::full({numClasses + 1, numPreds, numBatches}, 1), true);

  std::vector<fl::Variable> targetBoxes = {fl::Variable(
      Tensor::fromVector({4, numTargets, numBatches}, targetBoxesVec), false)};

  std::vector<fl::Variable> targetClasses = {fl::Variable(
      Tensor::fromVector({numTargets, numBatches}, targetClassVec), false)};
  auto matcher = HungarianMatcher(1, 1, 1);
  auto crit = SetCriterion(80, matcher, getLossWeights(), 0.0);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  EXPECT_EQ(loss["lossGiou_0"].scalar<float>(), 0.0);
}

TEST(SetCriterion, PytorchReproMultiplePreds) {
  // TODO: This should really be a fixture
  const int numClasses = 80;
  const int numTargets = 1;
  const int numPreds = 2;
  const int numBatches = 1;
  std::vector<float> predBoxesVec = {2, 2, 3, 3, 1, 1, 2, 2};

  std::vector<float> targetBoxesVec = {2, 2, 3, 3};

  std::vector<float> targetClassVec = {1};
  auto predBoxes = fl::Variable(
      Tensor::fromVector({4, numPreds, numBatches, 1}, predBoxesVec), true);
  auto predLogits =
      fl::Variable(fl::full({numClasses + 1, numPreds, numBatches}, 1), true);

  std::vector<fl::Variable> targetBoxes = {fl::Variable(
      Tensor::fromVector({4, numTargets, numBatches}, targetBoxesVec), false)};

  std::vector<fl::Variable> targetClasses = {fl::Variable(
      Tensor::fromVector({1, numTargets, numBatches}, targetClassVec), false)};
  auto matcher = HungarianMatcher(1, 1, 1);
  auto crit = SetCriterion(80, matcher, getLossWeights(), 0.0);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  EXPECT_EQ(loss["lossGiou_0"].scalar<float>(), 0.0);
}

TEST(SetCriterion, PytorchReproMultipleTargets) {
  const int numClasses = 80;
  const int numTargets = 2;
  const int numPreds = 2;
  const int numBatches = 1;
  std::vector<float> predBoxesVec = {2, 2, 3, 3, 1, 1, 2, 2};

  std::vector<float> targetBoxesVec = {
      1,
      1,
      2,
      2,
      2,
      2,
      3,
      3,
  };

  std::vector<float> targetClassVec = {1};
  auto predBoxes = fl::Variable(
      Tensor::fromVector({4, numPreds, numBatches, 1}, predBoxesVec), true);
  auto predLogits =
      fl::Variable(fl::full({numClasses + 1, numPreds, numBatches}, 1), true);

  std::vector<fl::Variable> targetBoxes = {fl::Variable(
      Tensor::fromVector({4, numTargets, numBatches}, targetBoxesVec), false)};

  std::vector<fl::Variable> targetClasses = {fl::Variable(
      Tensor::fromVector({numTargets, numBatches}, targetClassVec), false)};
  auto matcher = HungarianMatcher(1, 1, 1);
  auto crit = SetCriterion(80, matcher, getLossWeights(), 0.0);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  EXPECT_FLOAT_EQ(loss["lossGiou_0"].scalar<float>(), 0.0);
}

TEST(SetCriterion, PytorchReproNoPerfectMatch) {
  const int numClasses = 80;
  const int numTargets = 2;
  const int numPreds = 2;
  const int numBatches = 1;
  std::vector<float> predBoxesVec = {2, 2, 3, 3, 1, 1, 2, 2};

  std::vector<float> targetBoxesVec = {
      0.9, 0.8, 1.9, 1.95, 1.9, 1.95, 2.9, 2.95};

  // std::vector<float> predLogitsVec((numClasses + 1) * numPreds * numPreds,
  // 0.0);

  std::vector<float> targetClassVec = {1, 1};

  auto predBoxes = fl::Variable(
      Tensor::fromVector({4, numPreds, numBatches, 1}, predBoxesVec), true);
  auto predLogits =
      fl::Variable(fl::full({numClasses + 1, numPreds, numBatches}, 1), true);

  std::vector<fl::Variable> targetBoxes = {fl::Variable(
      Tensor::fromVector({4, numTargets, numBatches}, targetBoxesVec), false)};

  std::vector<fl::Variable> targetClasses = {fl::Variable(
      Tensor::fromVector({numTargets, numBatches}, targetClassVec), false)};
  auto matcher = HungarianMatcher(1, 1, 1);
  auto crit = SetCriterion(80, matcher, getLossWeights(), 0.0);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  EXPECT_FLOAT_EQ(loss["lossGiou_0"].scalar<float>(), 0.18111613);
  EXPECT_FLOAT_EQ(loss["lossBbox_0"].scalar<float>(), 0.3750);
}

TEST(SetCriterion, PytorchMismatch1) {
  const int numClasses = 80;
  const int numTargets = 1;
  const int numPreds = 1;
  const int numBatches = 1;
  std::vector<float> predBoxesVec = {
      2,
      2,
      3,
      3,
  };

  std::vector<float> targetBoxesVec1 = {
      1,
      1,
      2,
      2,
  };

  std::vector<float> targetClassVec = {1, 1};

  auto predBoxes = fl::Variable(
      Tensor::fromVector({4, numPreds, numBatches, 1}, predBoxesVec), true);
  auto predLogits =
      fl::Variable(fl::full({numClasses + 1, numPreds, numBatches}, 1), true);

  std::vector<fl::Variable> targetBoxes = {
      fl::Variable(
          Tensor::fromVector({4, numTargets, numPreds}, targetBoxesVec1),
          false),
  };

  std::vector<fl::Variable> targetClasses = {
      fl::Variable(
          Tensor::fromVector({1, numTargets, numPreds}, targetClassVec), false),
  };
  auto matcher = HungarianMatcher(1, 1, 1);
  auto crit = SetCriterion(80, matcher, getLossWeights(), 0.0);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  EXPECT_FLOAT_EQ(loss["lossGiou_0"].scalar<float>(), 0.91314667f);
  EXPECT_FLOAT_EQ(loss["lossBbox_0"].scalar<float>(), 4.f);
}

TEST(SetCriterion, PytorchMismatch2) {
  const int numClasses = 80;
  const int numTargets = 1;
  const int numPreds = 1;
  const int numBatches = 1;
  std::vector<float> predBoxesVec = {
      1,
      1,
      2,
      2,
  };

  std::vector<float> targetBoxesVec1 = {
      2,
      2,
      3,
      3,
  };

  std::vector<float> targetClassVec = {1, 1};

  auto predBoxes = fl::Variable(
      Tensor::fromVector({4, numPreds, numBatches, 1}, predBoxesVec), true);
  auto predLogits =
      fl::Variable(fl::full({numClasses + 1, numPreds, numBatches}, 1), true);

  std::vector<fl::Variable> targetBoxes = {
      fl::Variable(
          Tensor::fromVector({4, numTargets, numPreds}, targetBoxesVec1),
          false),
  };

  std::vector<fl::Variable> targetClasses = {
      fl::Variable(
          Tensor::fromVector({1, numTargets, numPreds}, targetClassVec), false),
  };
  auto matcher = HungarianMatcher(1, 1, 1);
  auto crit = SetCriterion(80, matcher, getLossWeights(), 0.0);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  EXPECT_FLOAT_EQ(loss["lossGiou_0"].scalar<float>(), 0.91314667f);
  EXPECT_FLOAT_EQ(loss["lossBbox_0"].scalar<float>(), 4.0f);
}

TEST(SetCriterion, PytorchReproBatching) {
  const int numClasses = 80;
  const int numTargets = 1;
  const int numPreds = 1;
  const int numBatches = 2;
  std::vector<float> predBoxesVec = {2, 2, 3, 3, 1, 1, 2, 2};

  std::vector<float> targetBoxesVec1 = {
      1,
      1,
      2,
      2,
  };

  std::vector<float> targetBoxesVec2 = {
      2,
      2,
      3,
      3,
  };

  std::vector<float> targetClassVec = {1, 1};

  auto predBoxes = fl::Variable(
      Tensor::fromVector({4, numPreds, numBatches, 1}, predBoxesVec), true);
  auto predLogits =
      fl::Variable(fl::full({numClasses + 1, numPreds, numBatches}, 1), true);

  std::vector<fl::Variable> targetBoxes = {
      fl::Variable(
          Tensor::fromVector({4, numTargets, numPreds}, targetBoxesVec1),
          false),
      fl::Variable(
          Tensor::fromVector({4, numTargets, numPreds}, targetBoxesVec2),
          false)};

  std::vector<fl::Variable> targetClasses = {
      fl::Variable(
          Tensor::fromVector({numTargets, numPreds, 1}, targetClassVec), false),
      fl::Variable(
          Tensor::fromVector({numTargets, numPreds, 1}, targetClassVec),
          false)};
  auto matcher = HungarianMatcher(1, 1, 1);
  auto crit = SetCriterion(80, matcher, getLossWeights(), 0.0);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  EXPECT_FLOAT_EQ(loss["lossGiou_0"].scalar<float>(), 0.91314667f);
  EXPECT_FLOAT_EQ(loss["lossBbox_0"].scalar<float>(), 4.f);
}

TEST(SetCriterion, DifferentNumberOfLabels) {
  const int numClasses = 80;
  const int numPreds = 2;
  const int numBatches = 2;
  std::vector<float> predBoxesVec = {
      2, 2, 3, 3, 1, 1, 2, 2, 2, 2, 3, 3, 1, 1, 2, 2};

  std::vector<float> targetBoxesVec1 = {
      1,
      1,
      2,
      2,
      2,
      2,
      3,
      3,
  };

  std::vector<float> targetBoxesVec2 = {
      2,
      2,
      3,
      3,
  };

  // std::vector<float> predLogitsVec((numClasses + 1) * numPreds * numPreds,
  // 0.0);

  std::vector<float> targetClassVec = {1, 1};

  auto predBoxes = fl::Variable(
      Tensor::fromVector({4, numPreds, numBatches, 1}, predBoxesVec), true);
  auto predLogits =
      fl::Variable(fl::full({numClasses + 1, numPreds, numBatches}, 1), true);

  std::vector<fl::Variable> targetBoxes = {
      fl::Variable(Tensor::fromVector({4, 2, 1}, targetBoxesVec1), false),
      fl::Variable(Tensor::fromVector({4, 1, 1}, targetBoxesVec2), false)};

  std::vector<fl::Variable> targetClasses = {
      fl::Variable(fl::full({2, 1, 1}, 1), false),
      fl::Variable(fl::full({1, 1, 1}, 1), false)};
  auto matcher = HungarianMatcher(1, 1, 1);
  auto crit = SetCriterion(80, matcher, getLossWeights(), 0.0);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  EXPECT_FLOAT_EQ(loss["lossGiou_0"].scalar<float>(), 0.f);
  EXPECT_FLOAT_EQ(loss["lossBbox_0"].scalar<float>(), 0.f);
}
// Test to make sure class labels are properly handles across batches
TEST(SetCriterion, DifferentNumberOfLabelsClass) {
  const int numClasses = 80;
  const int numPreds = 3;
  const int numBatches = 2;
  std::vector<float> predBoxesVec = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<float> targetBoxesVec1 = {1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<float> targetBoxesVec2 = {
      1,
      1,
      1,
      1,
  };

  auto predBoxes = fl::Variable(
      Tensor::fromVector({4, numPreds, numBatches, 1}, predBoxesVec), true);
  auto predLogitsT = fl::full({numClasses + 1, numPreds, numBatches}, 1.);
  predLogitsT(1, 1, 0) = 10; // These should get matched
  predLogitsT(2, 2, 0) = 10;
  predLogitsT(9, 1, 1) = 10;
  auto predLogits = fl::Variable(predLogitsT, true);

  std::vector<fl::Variable> targetBoxes = {
      fl::Variable(Tensor::fromVector({4, 2, 1}, targetBoxesVec1), false),
      fl::Variable(Tensor::fromVector({4, 1, 1}, targetBoxesVec2), false)};

  std::vector<fl::Variable> targetClasses = {
      fl::Variable(fl::iota({2}), false),
      fl::Variable(fl::full({1, 1, 1}, 9), false)};
  auto matcher = HungarianMatcher(1, 1, 1);
  auto crit = SetCriterion(80, matcher, getLossWeights(), 0.0);
  auto loss = crit.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  EXPECT_FLOAT_EQ(loss["lossGiou_0"].scalar<float>(), 0.f);
  EXPECT_FLOAT_EQ(loss["lossBbox_0"].scalar<float>(), 0.f);
  EXPECT_NEAR(loss["lossCe_0"].scalar<float>(), 1.4713663f, 1e-4);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
