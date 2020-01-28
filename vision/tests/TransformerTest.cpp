#include "vision/nn/Transformer.h"

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::cv;

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
      0.0,
      0);

  ASSERT_EQ(result.scalar<float>(), 2.0);
}

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
      0.0,
      0);

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
      0.0,
      0);

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
      0.0,
      0);

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
  Transformer tr(C, numHeads, numEncoderDecoder, numEncoderDecoder, mlpDim, dropout);

  std::vector<Variable> inputs = { 
    Variable(af::randu(W, H, C, B), false), // src
    Variable(af::randu(af::dim4(C, bbox_queries)), false)
  };
  auto output = tr(inputs)[0];
  ASSERT_EQ(output.dims(0), C) << "Transformer should return model dim as first dimension";
  ASSERT_EQ(output.dims(1), bbox_queries) << "Transformer did not return the correct number of labels";
  ASSERT_EQ(output.dims(2), B) << "Transformer did not return the correct number of batches";

}
