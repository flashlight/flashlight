#include <stdio.h>
#include "Halide.h"
using namespace Halide;

int main(int argc, char** argv) {
  // We'll define a simple one-stage pipeline:
  Func brighter;
  Var x, y;

  // The pipeline will depend on one scalar parameter.
  Param<uint8_t> offset;

  // And take one grayscale 8-bit input buffer. The first
  // constructor argument gives the type of a pixel, and the second
  // specifies the number of dimensions (not the number of
  // channels!). For a grayscale image this is two; for a color
  // image it's three. Currently, four dimensions is the maximum for
  // inputs and outputs.
  ImageParam input(type_of<uint8_t>(), 2);

  // If we were jit-compiling, these would just be an int and a
  // Buffer, but because we want to compile the pipeline once and
  // have it work for any value of the parameter, we need to make a
  // Param object, which can be used like an Expr, and an ImageParam
  // object, which can be used like a Buffer.

  // Define the Func.
  brighter(x, y) = input(x, y) + offset;

  // Schedule it.
  brighter.vectorize(x, 16).parallel(y);

  // This time, instead of calling brighter.realize(...), which
  // would compile and run the pipeline immediately, we'll call a
  // method that compiles the pipeline to a static library and header.
  //
  // For AOT-compiled code, we need to explicitly declare the
  // arguments to the routine. This routine takes two. Arguments are
  // usually Params or ImageParams.
  brighter.compile_to_static_library(
      "HalideTestPipeline", {input, offset}, "brighter");

  printf("Halide pipeline compiled, but not yet run.\n");

  // To continue this lesson, look in the file lesson_10_aot_compilation_run.cpp

  return 0;
}
