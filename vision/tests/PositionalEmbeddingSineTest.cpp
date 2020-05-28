#include "vision/nn/PositionalEmbeddingSine.h"

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::cv;

TEST(PositionalEmbeddingSine, Test1) {

  int hiddenDim = 8;
  int H = 6;
  int W = 3;
  int C = 64;
  int B = 1;
  af::dim4 dims = { W, H, C, B };
  auto input = Variable(af::constant(1, dims), false);
  PositionalEmbeddingSine pos(hiddenDim/2, 10000.0f, false, 0.0f);

  auto result = pos.forward(input);
  std::cout << result.dims() << std::endl;
  std::cout << result(0, 5, 3, 0).scalar<float>() << std::endl;
  std::cout << result(0, 0, 0, 0).scalar<float>() << std::endl;
  af_print(result.array());
}
