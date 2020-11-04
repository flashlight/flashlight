#include "app/object_detection/nn/Transformer.h"
#include "app/object_detection/nn/PositionalEmbeddingSine.h"

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::app::object_detection;

TEST(Tranformer, BasicAttention) {
  int B = 1;
  int S = 5;
  int E = 1;
  int L = 1;
  int nHeads = 1;

  auto keys = af::constant(0.0, {E, B, S});
  keys(0, 0, 2) = 10000;


  auto query = Variable(af::constant(10000, { E, B, L }), false);
  auto key = Variable(keys, false);
  auto value = Variable(af::iota({ E, B, S }), false);


  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
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

  auto keys = af::constant(0.0, {E, B, S});
  keys(0, 0, 2) = 10000;
  keys(0, 0, 4) = 10000;


  auto query = Variable(af::constant(10000, { E, B, L }), false);
  auto key = Variable(keys, false);
  auto value = Variable(af::iota({ E, B, S }), false);


  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
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

  auto keys = af::constant(0.0, {E, B, S});
  keys(0, 0, 2) = 10000;
  keys(0, 0, 4) = 10000;


  auto query = Variable(af::constant(10000, { E, B, L }), false);
  auto key = Variable(keys, false);
  auto value = Variable(af::iota({ E, B, S }), false);
  int maskLength = 3;
  auto maskArray = af::constant(0, { S, B });
  maskArray(af::seq(0, maskLength - 1), af::span) = af::constant(1, { maskLength, B });
  auto mask = Variable(maskArray, false);


  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
      mask,
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

  auto keys = af::constant(0.0, {E, B, S});
  keys(0, 0, 2) = 10000; // First head --> 2
  keys(1, 0, 3) = 10000;// Second head attend to 3


  auto query = Variable(af::constant(10000, { E, B, L }), false);
  auto key = Variable(keys, false);
  //auto value = Variable(af::iota({ E, B, S }), false);
  auto value = Variable(af::iota({ 1, 1, S }, {E, B, 1}), false);


  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
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

  auto keys = af::constant(0.0, {E, B, S});
  keys(0, 0, 2) = 10000; // First head --> 2
  keys(1, 0, 3) = 10000;// Second head attend to 3
  keys(0, 1, 1) = 10000; // First head --> 2
  keys(1, 1, 3) = 10000;// Second head attend to 3


  auto query = Variable(af::constant(10000, { E, B, L }), false);
  auto key = Variable(keys, false);
  auto value = Variable(af::iota({ 1, 1, S }, {E, B, 1}), false);


  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
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

  auto keys = af::constant(0.0, {E, B, S});
  keys(0, 0, 2) = 10000; 
  keys(0, 0, 1) = -10000; 
  // Second head
  keys(1, 0, 3) = -10000;
  keys(1, 0, 0) = 10000; 

  auto queries = af::constant(0.0, {E, B, L});
  queries(0, 0, 0) = 10000;// Matches with 2
  queries(1, 0, 0) = -10000; // matches with 3
  // Second query
  queries(0, 0, 1) = -10000; // Matches with 1
  queries(1, 0, 1) = 10000; // matches 0


  auto query = Variable(queries, false);
  auto key = Variable(keys, false);
  auto value = Variable(af::iota({ 1, 1, S }, {E, B, 1}), false);


  auto result = transformerMultiheadAttention(
      query,
      key,
      value,
      Variable(),
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
  fl::app::object_detection::Transformer tr(C, numHeads, numEncoderDecoder, numEncoderDecoder, mlpDim, dropout);

  std::vector<Variable> inputs = { 
    Variable(af::randu(W, H, C, B), false), // input Projection
    Variable(af::randu(af::dim4(W, H, 1, B)), false), // mask 
    Variable(af::randu(af::dim4(C, bbox_queries)), false), // query_embed 
    Variable(af::randu(af::dim4(W, H, C, B)), false) // query_embed 
  };
  auto output = tr(inputs)[0];
  ASSERT_EQ(output.dims(0), C) << "Transformer should return model dim as first dimension";
  ASSERT_EQ(output.dims(1), bbox_queries) << "Transformer did not return the correct number of labels";
  ASSERT_EQ(output.dims(2), B) << "Transformer did not return the correct number of batches";

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
  fl::app::object_detection::Transformer tr(C, numHeads, numEncoderDecoder, numEncoderDecoder, mlpDim, dropout);

  PositionalEmbeddingSine pos(C/2, 10000.0f, false, 0.0f);

  auto actualDims = af::dim4(maskW, maskH, 1, B);
  auto nonMask = af::constant(1, actualDims);

  auto maskArray = af::constant(0, { W, H, 1, B });
  maskArray(af::seq(0, maskW - 1), af::seq(0, maskH - 1), af::span, af::span) = nonMask;
  auto mask = Variable(maskArray, false);
  auto nonMaskPos = pos.forward(Variable(nonMask, false));

  std::vector<Variable> nonMaskInput = { 
    Variable(af::randu(maskW, maskH, C, B), false), // input Projection
    Variable(af::constant(1, af::dim4(maskW, maskH, 1, B)), false), // mask 
    Variable(af::randu(af::dim4(C, bbox_queries)), false), // query_embed 
    nonMaskPos
  };
  auto nonMaskOutput = tr(nonMaskInput)[0];
  std::cout << "Here" << std::endl;

  auto nonMaskedSrc = af::randu(W, H, C, B);
  nonMaskedSrc(af::seq(0, maskW - 1), af::seq(0, maskH - 1), af::span, af::span) = nonMaskInput[0].array();

  auto maskPos = pos.forward(mask);

  std::vector<Variable> maskInput = { 
    Variable(nonMaskedSrc, false), // input Projection
    mask, // mask 
    nonMaskInput[2], // query_embed 
    maskPos
  };
  auto maskOutput = tr(maskInput)[0];
  af_print(nonMaskPos.array());
  af_print(maskPos.array());
  af_print(nonMaskOutput.array());
  af_print(maskOutput.array());
  //ASSERT_EQ(output.dims(0), C) << "Transformer should return model dim as first dimension";
  //ASSERT_EQ(output.dims(1), bbox_queries) << "Transformer did not return the correct number of labels";
  //ASSERT_EQ(output.dims(2), B) << "Transformer did not return the correct number of batches";
}
