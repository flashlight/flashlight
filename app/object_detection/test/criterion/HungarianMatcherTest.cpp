#include "vision/criterion/Hungarian.h"
#include "vision/criterion/SetCriterion.h"

#include "flashlight/autograd/Variable.h"
#include "flashlight/flashlight.h"

#include <vector>
#include <af/array.h>

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::cv;

/*
TEST(HungarianMatcher, Test1) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 2;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 1;
  std::vector<float> predBoxesVec = {
    2, 2, 3, 3,
    1, 1, 2, 2
  };

  std::vector<float> targetBoxesVec = {
    1, 1, 2, 2,
    2, 2, 3, 3
  };

  std::vector<float> predLogitsVec = {
    1, 2,
    2, 1
  };

  std::vector<float> targetClassVec = {
    0, 1
  };
  std::vector<float> expRowIds = {
    0, 1
  };
  std::vector<float> colRowIds = {
    1, 0
  };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(1, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto matcher = HungarianMatcher(1, 1, 1);
  auto costs = matcher.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  auto rowIds = costs[0].first;
  auto colIds = costs[0].second;
  af_print(colIds);
  for(int i = 0; i < NUM_PREDS; i++) {
    EXPECT_EQ(colIds(i).scalar<int>(), colRowIds[i]);
  }
}

TEST(HungarianMatcher, Test2) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 1;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 1;
  std::vector<float> predBoxesVec = {
    2, 2, 3, 3,
    1, 1, 2, 2
  };

  std::vector<float> targetBoxesVec = { 1, 1, 2, 2, };
  std::vector<float> predLogitsVec = { 2, 1 };
  std::vector<float> targetClassVec = { 0 };
  std::vector<float> expRowIds = { 0 };
  std::vector<float> colRowIds = { 1 };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(1, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto matcher = HungarianMatcher(1, 1, 1);
  auto costs = matcher.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  auto rowIds = costs[0].first;
  auto colIds = costs[0].second;
  for(int i = 0; i < NUM_TARGETS; i++) {
    EXPECT_EQ(colIds(i).scalar<int>(), colRowIds[i]);
  }
  af_print(colIds);
  af_print(rowIds);
}

TEST(HungarianMatcher, Test3) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 1;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 1;
  std::vector<float> predBoxesVec = {
    1, 1, 2, 2,
    2, 2, 3, 3
  };

  std::vector<float> targetBoxesVec = { 1, 1, 2, 2, };
  std::vector<float> predLogitsVec = { 2, 1 };
  std::vector<float> targetClassVec = { 0 };
  std::vector<float> expRowIds = { 0 };
  std::vector<float> colRowIds = { 0 };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(1, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto matcher = HungarianMatcher(1, 1, 1);
  auto costs = matcher.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  auto rowIds = costs[0].first;
  auto colIds = costs[0].second;
  for(int i = 0; i < NUM_TARGETS; i++) {
    EXPECT_EQ(colIds(i).scalar<int>(), colRowIds[i]);
  }
  af_print(colIds);
  af_print(rowIds);
}

TEST(HungarianMatcher, TestBatching) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 2;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 2;
  std::vector<float> predBoxesVec = {
    1, 1, 2, 2,
    2, 2, 3, 3,
    2, 2, 3, 3,
    1, 1, 2, 2
  };

  std::vector<float> targetBoxesVec = { 
    1, 1, 2, 2, 
    2, 2, 3, 3,
    1, 1, 2, 2, 
    2, 2, 3, 3,
  };
  std::vector<float> predLogitsVec = { 2, 1, 2, 1 };
  std::vector<float> targetClassVec = { 0, 0, 0, 0 };
  std::vector<float> expRowIds = { 0, 1, 0, 1 };
  std::vector<float> colRowIds = { 0, 1, 1, 0 };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(1, NUM_TARGETS, NUM_BATCHES, targetClassVec.data());
  auto matcher = HungarianMatcher(1, 1, 1);
  auto costs = matcher.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  for(int b = 0; b < NUM_BATCHES; b++) {
    auto colIds = costs[b].second;
    for(int i = 0; i < NUM_TARGETS; i++) {
      EXPECT_EQ(colIds(i).scalar<int>(), colRowIds[i + b * NUM_TARGETS]);
    }
  }
}

TEST(HungarianMatcher, TestClass) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 2;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 2;
  std::vector<float> predBoxesVec = {
    1, 1, 2, 2, // 1
    1, 1, 2, 2, // 0
    1, 1, 2, 2, // 0
    1, 1, 2, 2 // 1
  };

  std::vector<float> targetBoxesVec = { 
    1, 1, 2, 2, // 0
    1, 1, 2, 2, // 1
    1, 1, 2, 2, // 0
    1, 1, 2, 2 // 1
  };
  std::vector<float> predLogitsVec = { 
    1, 2,  // Class 2
    2, 1,  // Class 1
    2, 1,  // Class 2
    1, 2  // Class 1
  };
  std::vector<float> targetClassVec = { 0, 1 , 0, 1};
  std::vector<float> expRowIds = { 0, 1, 0, 1 };
  std::vector<float> colRowIds = { 1, 0, 0, 1 };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(1, NUM_TARGETS, NUM_BATCHES, targetClassVec.data());
  auto matcher = HungarianMatcher(1, 1, 1);
  auto costs = matcher.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  for(int b = 0; b < NUM_BATCHES; b++) {
    auto colIds = costs[b].second;
    af_print(colIds);
    for(int i = 0; i < NUM_TARGETS; i++) {
      EXPECT_EQ(colIds(i).scalar<int>(), colRowIds[i + b * NUM_TARGETS]) << "i" << i << " b" << b << std::endl;
    }
  }
}
*/

TEST(HungarianMatcher, TestClass) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 3;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 2;
  std::vector<float> predBoxesVec = {
    1, 1, 2, 2, // 1
    1, 1, 2, 2, // 0
    1, 1, 2, 2, // 0
    1, 1, 2, 2 // 1
  };

  std::vector<float> targetBoxesVec = { 
    1, 1, 2, 2, // 0
    1, 1, 2, 2, // 1
    -1, -1, -1, -1, // 1
    1, 1, 2, 2, // 0
    1, 1, 2, 2 // 1
    -1, -1, -1, -1, // 1
  };
  std::vector<float> targetLensVec = { 
   2, 2
  };
  std::vector<float> predLogitsVec = { 
    1, 2,  // Class 2
    2, 1,  // Class 1
    2, 1,  // Class 2
    1, 2  // Class 1
  };
  std::vector<float> targetClassVec = { 0, 1 , -1, 0, 1, -1};
  std::vector<float> expRowIds = { 0, 1, 0, 1 };
  std::vector<float> colRowIds = { 1, 0, 0, 1 };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predLogitsVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(1, NUM_TARGETS, NUM_BATCHES, targetClassVec.data());
  auto targetLens = af::array(NUM_BATCHES, targetLensVec.data());
  auto matcher = HungarianMatcher(1, 1, 1);
  auto costs = matcher.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  //for(int b = 0; b < NUM_BATCHES; b++) {
    //auto colIds = costs[b].second;
    //af_print(colIds);
    //for(int i = 0; i < NUM_TARGETS; i++) {
      //EXPECT_EQ(colIds(i).scalar<int>(), colRowIds[i + b * NUM_TARGETS]) << "i" << i << " b" << b << std::endl;
    //}
  //}
}




