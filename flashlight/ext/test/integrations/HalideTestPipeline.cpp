#include <iostream>

#include "Halide.h"

using namespace Halide;

int main(int argc, char** argv) {
  // We'll define a simple one-stage pipeline:
  Func brighter;
  Var x, y;

  // The pipeline will depend on one scalar parameter.
  Param<float> offset;

  // ctor is pixel element type, number of dimensions
  ImageParam input(type_of<float>(), 2);

  // If we were jit-compiling, these would just be an int and a
  // Buffer, but because we want to compile the pipeline once and
  // have it work for any value of the parameter, we need to make a
  // Param object, which can be used like an Expr, and an ImageParam
  // object, which can be used like a Buffer.

  // Define the Func.
  brighter(x, y) = input(x, y) + offset;

  Var x_outer, y_outer, x_inner, y_inner;
  brighter.gpu_tile(
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
  // brighter.vectorize(x, 16).parallel(y);

  // This time, instead of calling brighter.realize(...), which
  // would compile and run the pipeline immediately, we'll call a
  // method that compiles the pipeline to a static library and header.
  //
  // For AOT-compiled code, we need to explicitly declare the
  // arguments to the routine. This routine takes two. Arguments are
  // usually Params or ImageParams.
  brighter.compile_to_static_library(
      "HalideTestPipeline",
      {input, offset},
      "brighter",
      Halide::get_target_from_environment()
          .with_feature(Halide::Target::Feature::CUDA)
          .with_feature(Halide::Target::Debug));

  std::cout << "HalideTestPipeline pipeline compiled, but not yet run."
            << std::endl;

  return 0;
}
