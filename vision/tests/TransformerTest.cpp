#include "vision/nn/Transformer.h"

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::cv;

TEST(Tranformer, Size) {
  int B = 2;
  int H = 3;
  int W = 3;
  int C = 16;
  float dropout = 0.5;
  int bbox_queries = 100;
  int numEncoderDecoder = 100;
  int mlpDim = 32;
  int numHeads = 6;
  Transformer tr(C, numHeads, numEncoderDecoder, numEncoderDecoder, mlpDim, dropout);

  std::vector<Variable> inputs = { 
    Variable(af::randu(W, H, C, B), false),
    Variable(af::randu(af::dim4(C, bbox_queries)), false)
  };
  auto output = tr(inputs)[0];
  af_print(output.array());
  ASSERT_EQ(output.dims(0), C) << "Transformer should return model dim as first dimension";
  ASSERT_EQ(output.dims(1), bbox_queries) << "Transformer did not return the correct number of labels";
  ASSERT_EQ(output.dims(2), B) << "Transformer did not return the correct number of batches";

}
