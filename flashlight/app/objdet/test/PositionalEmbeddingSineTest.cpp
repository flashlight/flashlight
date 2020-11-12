#include "flashlight/app/objdet/nn/PositionalEmbeddingSine.h"

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::app::object_detection;

TEST(PositionalEmbeddingSine, Test1) {

  int hiddenDim = 8;
  int H = 6;
  int W = 6;
  int B = 1;
  af::dim4 dims = { W, H, 1, B };
  auto inputArray = af::constant(0, dims);
  inputArray(af::seq(0, 3), af::seq(0, 3)) = af::constant(1, {4, 4, B});
  auto input = Variable(inputArray, false);

  PositionalEmbeddingSine pos(hiddenDim/2, 10000.0f, false, 0.0f);

  auto result = pos.forward(input);
  std::cout << result.dims() << std::endl;
  std::cout << result(0, 5, 3, 0).scalar<float>() << std::endl;
  std::cout << result(0, 0, 0, 0).scalar<float>() << std::endl;
  af_print(result.array());
}
