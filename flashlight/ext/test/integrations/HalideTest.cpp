/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/ext/integrations/halide/HalideInterface.h"
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"

// Generated at build time -- see the accompanying CMakeList
#include "HalideTestPipeline.h"

using namespace fl;

TEST(HalideTest, ConvertVariableAndRun) {
  int yDim = 20, xDim = 20;
  auto inputVar =
      Variable((af::randu({yDim, xDim}) * 100).as(af::dtype::u8), false);
  auto outputVar = Variable(af::array({yDim, xDim}, af::dtype::u8), false);

  // Runtime:Buffer's
  auto input = toHalideBuffer<uint8_t>(inputVar);
  auto output = toHalideBuffer<uint8_t>(outputVar);

  int offset = 5;
  int error = brighter(*input.get(), offset, *output.get());

  EXPECT_FALSE(error) << "Halide returned error.";

  // Now let's check the filter performed as advertised. It was
  // supposed to add the offset to every input pixel.
  for (int y = 0; y < yDim; y++) {
    for (int x = 0; x < xDim; x++) {
      uint8_t input_val = input(x, y);
      uint8_t output_val = output(x, y);
      uint8_t correct_val = input_val + offset;
      EXPECT_EQ(output_val, correct_val);
    }
  }
  // Ensure the output buffers is correctly-populated
  af::print("in", inputVar.array());
  af::print("out", outputVar.array());
  EXPECT_TRUE(allClose(inputVar.array() + offset, outputVar.array()));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
