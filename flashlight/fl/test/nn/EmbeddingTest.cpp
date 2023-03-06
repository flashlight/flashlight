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

TEST(ModuleTest, EmbeddingFwd) {
  int embDim = 3, nEmb = 5, nQuery = 2, batchSize = 2;
  auto wtVar = param(Tensor::fromVector<float>(
      {embDim, nEmb}, {8, 2, 2, 10, 5, 3, 3, 4, 6, 12, 3, 8, 0, 5, 2}));

  auto inVar = input(Tensor::fromVector<float>({2, batchSize}, {1, 3, 0, 0}));

  auto expectedOutVar = Variable(
      Tensor::fromVector<float>(
          {embDim, nQuery, batchSize}, {10, 5, 3, 12, 3, 8, 8, 2, 2, 8, 2, 2}),
      true);

  // Var initialization
  auto emb = Embedding(wtVar);
  ASSERT_TRUE(allClose(emb.forward(inVar), expectedOutVar, 1E-7));

  // Regular initialization
  emb = Embedding(embDim, nEmb);
  wtVar = emb.param(0);
  ASSERT_EQ(wtVar.shape(), Shape({embDim, nEmb}));

  expectedOutVar = Variable(
      fl::reshape(
          wtVar.tensor()(fl::span, inVar.tensor()),
          {embDim, nQuery, batchSize}),
      true);
  ASSERT_TRUE(allClose(emb.forward(inVar), expectedOutVar, 1E-7));
}
