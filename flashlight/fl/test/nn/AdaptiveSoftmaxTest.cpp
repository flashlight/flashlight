/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>

#include <gtest/gtest.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/common.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

TEST(ModuleTest, AdaptiveSoftMaxPredict) {
  // test predict gives the same as argmax along probs
  int N = 5;
  int C = 5;
  int T = 10;
  int B = 5;

  auto x = input(fl::rand({N, T, B}, fl::dtype::f32));
  auto y = Variable(
      (fl::rand({T, B}, fl::dtype::u32) % C).astype(fl::dtype::s32), false);

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);

  auto probs = activation->forward(x);
  auto result1 = activation->predict(x).tensor();
  auto result2 = fl::argmax(probs.tensor(), 0, /* keepDims = */ true);

  ASSERT_TRUE(allClose(result1, result2));
}

TEST(ModuleTest, AdaptiveSoftMaxLossBatchFwd) {
  // test batching
  int N = 5;
  int C = 3;
  int T = 10;
  int B = 5;

  auto x = input(fl::rand({N, T, B}, fl::dtype::f32));
  auto y = Variable(
      (fl::rand({T, B}, fl::dtype::u32) % C).astype(fl::dtype::s32), false);

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);
  auto asml =
      std::make_shared<AdaptiveSoftMaxLoss>(activation, ReduceMode::NONE);
  auto batchOutVar = asml->forward(x, y);

  auto singleOut = fl::full({T, B}, 0, fl::dtype::f32);
  for (int i = 0; i < B; ++i) {
    auto singleOutVar = asml->forward(
        x(fl::span, fl::span, fl::range(i, i + 1)), y(fl::span, fl::range(i, i + 1)));
    singleOut(fl::span, i) = singleOutVar.tensor();
  }

  ASSERT_TRUE(allClose(batchOutVar.tensor(), singleOut));
}

TEST(ModuleTest, AdaptiveSoftMaxLossIgnoreIndex) {
  // test batching
  int N = 5;
  int C = 3;
  int T = 10;
  int B = 5;

  auto x = input(fl::rand({N, T, B}, fl::dtype::f32));
  auto y = Variable(
      (fl::rand({T, B}, fl::dtype::u32) % C).astype(fl::dtype::s32), false);
  auto ignoreIdx = y(0, 0).scalar<int>();
  auto ignoreCount = fl::sum(y.tensor() != ignoreIdx).scalar<unsigned>();

  std::vector<int> cutoff{{C / 2, C}};
  auto activation = std::make_shared<AdaptiveSoftMax>(N, cutoff);
  auto asml1 = std::make_shared<AdaptiveSoftMaxLoss>(
      activation, ReduceMode::SUM, ignoreIdx);
  auto asml2 = std::make_shared<AdaptiveSoftMaxLoss>(
      activation, ReduceMode::MEAN, ignoreIdx);

  auto lossSum = asml1->forward(x, y);
  auto lossMean = asml2->forward(x, y);
  ASSERT_NEAR(
      fl::sum(lossSum.tensor()).scalar<float>(),
      fl::sum(lossMean.tensor()).scalar<float>() * ignoreCount,
      1E-5);
}
