/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/nn/Transformer.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/pkg/vision/nn/PositionalEmbeddingSine.h"

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::pkg::vision;

TEST(Tranformer, BasicAttention) {
  int B = 1;
  int S = 5;
  int E = 1;
  int L = 1;
  int nHeads = 1;

  auto keys = fl::full({E, B, S}, 0.);
  keys(0, 0, 2) = 10000;

  auto query = Variable(fl::full({E, B, L}, 10000, fl::dtype::f32), false);
  auto key = Variable(keys, false);
  auto value = Variable(fl::iota({E, B, S}, {}, fl::dtype::f32), false);

  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
      nHeads, // num_heads
      0.0);

  ASSERT_EQ(result.scalar<float>(), 2.0);
};

TEST(Tranformer, BasicAttentionNonMasked) {
  int B = 1;
  int S = 5;
  int E = 1;
  int L = 1;
  int nHeads = 1;

  auto keys = fl::full({E, B, S}, 0.);
  keys(0, 0, 2) = 10000;
  keys(0, 0, 4) = 10000;

  auto query = Variable(fl::full({E, B, L}, 10000, fl::dtype::f32), false);
  auto key = Variable(keys, false);
  auto value = Variable(fl::iota({E, B, S}, {}, fl::dtype::f32), false);

  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
      nHeads, // num_heads
      0.0);

  ASSERT_EQ(result.scalar<float>(), 3.0);
};

TEST(Tranformer, BasicAttentionMasked) {
  int B = 1;
  int S = 5;
  int E = 1;
  int L = 1;
  int nHeads = 1;

  auto keys = fl::full({E, B, S}, 0.);
  keys(0, 0, 2) = 10000;
  keys(0, 0, 4) = 10000;

  auto query = Variable(fl::full({E, B, L}, 10000, fl::dtype::f32), false);
  auto key = Variable(keys, false);
  auto value = Variable(fl::iota({E, B, S}, {}, fl::dtype::f32), false);
  int maskLength = 3;
  auto mask = fl::full({S, B}, 0);
  mask(fl::range(0, maskLength)) = fl::full({maskLength, B}, 1);

  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(mask, false),
      nHeads, // num_heads
      0.0);

  ASSERT_EQ(result.scalar<float>(), 2.0);
};

TEST(Tranformer, MultiHeadedAttention) {
  int B = 1;
  int S = 5;
  int E = 2;
  int L = 1;
  int nHeads = 2;

  auto keys = fl::full({E, B, S}, 0.);
  keys(0, 0, 2) = 10000; // First head --> 2
  keys(1, 0, 3) = 10000; // Second head attend to 3

  auto query = Variable(fl::full({E, B, L}, 10000, fl::dtype::f32), false);
  auto key = Variable(keys, false);
  // auto value = Variable(fl::iota({ E, B, S }), false);
  auto value = Variable(fl::iota({1, 1, S}, {E, B, 1}), false);

  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
      nHeads, // num_heads
      0.0);

  ASSERT_EQ(result(0).scalar<float>(), 2.0f);
  ASSERT_EQ(result(1).scalar<float>(), 3.0f);
}

TEST(Tranformer, MultiHeadedAttentionBatch) {
  int B = 2;
  int S = 5;
  int E = 2;
  int L = 1;
  int nHeads = 2;

  auto keys = fl::full({E, B, S}, 0.);
  keys(0, 0, 2) = 10000; // First head --> 2
  keys(1, 0, 3) = 10000; // Second head attend to 3
  keys(0, 1, 1) = 10000; // First head --> 2
  keys(1, 1, 3) = 10000; // Second head attend to 3

  auto query = Variable(fl::full({E, B, L}, 10000, fl::dtype::f32), false);
  auto key = Variable(keys, false);
  auto value = Variable(fl::iota({1, 1, S}, {E, B, 1}), false);

  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
      nHeads, // num_heads
      0.0);

  ASSERT_EQ(result(0, 0).scalar<float>(), 2.0f);
  ASSERT_EQ(result(1, 0).scalar<float>(), 3.0f);
  ASSERT_EQ(result(0, 1).scalar<float>(), 1.0f);
  ASSERT_EQ(result(1, 1).scalar<float>(), 3.0f);
}

TEST(Tranformer, MultiHeadedAttentionMultipleQueries) {
  int B = 1;
  int S = 5;
  int E = 2;
  int L = 2;
  int nHeads = 2;

  auto keys = fl::full({E, B, S}, 0.);
  keys(0, 0, 2) = 10000;
  keys(0, 0, 1) = -10000;
  // Second head
  keys(1, 0, 3) = -10000;
  keys(1, 0, 0) = 10000;

  auto queries = fl::full({E, B, L}, 0.);
  queries(0, 0, 0) = 10000; // Matches with 2
  queries(1, 0, 0) = -10000; // matches with 3
  // Second query
  queries(0, 0, 1) = -10000; // Matches with 1
  queries(1, 0, 1) = 10000; // matches 0

  auto query = Variable(queries, false);
  auto key = Variable(keys, false);
  auto value = Variable(fl::iota({1, 1, S}, {E, B, 1}), false);

  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
      nHeads, // num_heads
      0.0);

  ASSERT_EQ(result(0, 0, 0).scalar<float>(), 2.0f);
  ASSERT_EQ(result(1, 0, 0).scalar<float>(), 3.0f);
  // Second query
  ASSERT_EQ(result(0, 0, 1).scalar<float>(), 1.0f);
  ASSERT_EQ(result(1, 0, 1).scalar<float>(), 0.0f);
}

TEST(Tranformer, Size) {
  int B = 3;
  int H = 5;
  int W = 5;
  int C = 16;
  float dropout = 0.5;
  int bbox_queries = 100;
  int numEncoderDecoder = 2;
  int mlpDim = 32;
  int numHeads = 8;
  fl::pkg::vision::Transformer tr(
      C, numHeads, numEncoderDecoder, numEncoderDecoder, mlpDim, dropout);

  std::vector<Variable> inputs = {
      Variable(fl::rand({W, H, C, B}), false), // input Projection
      Variable(fl::rand({W, H, 1, B}), false), // mask
      Variable(fl::rand({C, bbox_queries}), false), // query_embed
      Variable(fl::rand({W, H, C, B}), false) // query_embed
  };
  auto output = tr(inputs)[0];
  ASSERT_EQ(output.dim(0), C)
      << "Transformer should return model dim as first dimension";
  ASSERT_EQ(output.dim(1), bbox_queries)
      << "Transformer did not return the correct number of labels";
  ASSERT_EQ(output.dim(2), B)
      << "Transformer did not return the correct number of batches";
}

TEST(Tranformer, Masked) {
  int B = 2;
  int H = 8;
  int W = 8;
  int maskH = 3;
  int maskW = 3;
  int C = 16;
  float dropout = 0.0;
  int bbox_queries = 2;
  int numEncoderDecoder = 2;
  int mlpDim = 32;
  int numHeads = 8;
  int hiddenDim = 8;
  fl::pkg::vision::Transformer tr(
      C, numHeads, numEncoderDecoder, numEncoderDecoder, mlpDim, dropout);

  PositionalEmbeddingSine pos(C / 2, 10000.0f, false, 0.0f);

  auto nonMask = fl::full({maskW, maskH, 1, B}, 1);

  auto mask = fl::full({W, H, 1, B}, 0);
  mask(fl::range(0, maskW), fl::range(0, maskH)) = nonMask;
  auto nonMaskPos = pos.forward({Variable(nonMask, false)})[0];

  std::cout << "--- nonMaskPos " << nonMaskPos.shape() << std::endl;

  std::vector<Variable> nonMaskInput = {
      Variable(fl::rand({maskW, maskH, C, B}), false), // input Projection
      Variable(fl::full({maskW, maskH, 1, B}, 1), false), // mask
      Variable(fl::rand({C, bbox_queries}), false), // query_embed
      nonMaskPos};
  auto nonMaskOutput = tr(nonMaskInput)[0];

  auto nonMaskedSrc = fl::rand({W, H, C, B});
  nonMaskedSrc(fl::range(0, maskW), fl::range(0, maskH)) =
      nonMaskInput[0].tensor();

  auto maskPos = pos.forward({fl::Variable(mask, false)})[0];

  std::vector<Variable> maskInput = {
      Variable(nonMaskedSrc, false), // input Projection
      Variable(mask, false), // mask
      nonMaskInput[2], // query_embed
      maskPos};
  auto maskOutput = tr(maskInput)[0];
}
