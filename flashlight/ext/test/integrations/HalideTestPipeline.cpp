#include <iostream>

#include "Halide.h"

using namespace Halide;

int main(int argc, char** argv) {
  Func testFunc;
  Var x, y;

  // Depends on a scalar param
  Param<float> offset;

  // ctor is element type, number of dimensions
  ImageParam input(type_of<float>(), 2);

  Func sinVals;
  sinVals(x, y) = sin(x * y);
  sinVals.compute_root(); // ensure we allocate memory in the test

  testFunc(x, y) = input(x, y) + sinVals(x, y) + offset;

  Var x_outer, y_outer, x_inner, y_inner;
  testFunc.gpu_tile(
      x,
      y,
      x_outer,
      y_outer,
      x_inner,
      y_inner,
      16, // blocks
      16, // threads
      Halide::TailStrategy::Auto,
      Halide::DeviceAPI::CUDA);

  testFunc.compile_to_static_library(
      "HalideTestPipeline",
      {input, offset}, // arguments
      "testFunc",
      Halide::get_target_from_environment()
          .with_feature(Halide::Target::Feature::CUDA)
          .with_feature(Halide::Target::Debug));

  std::cout << "HalideTestPipeline pipeline compiled, but not yet run."
            << std::endl;

  return 0;
}
