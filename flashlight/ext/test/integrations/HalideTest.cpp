/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TODO remove
#include <cuda_runtime.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "flashlight/ext/integrations/halide/HalideInterface.h"
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"

// Generated at build time -- see the accompanying CMakeList
#include "HalideTestPipeline.h"

using namespace fl;

TEST(HalideTest, TypeMapping) {
  Halide::Buffer<float> floatBuf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(floatBuf.type()), af::dtype::f32);

  Halide::Buffer<uint32_t> intBuf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(intBuf.type()), af::dtype::u32);

  Halide::Buffer<int> uintBuf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(uintBuf.type()), af::dtype::s32);
}

TEST(HalideTest, ConvertDims) {
  af::dim4 dims(1, 5, 3, 6);
  EXPECT_THAT(ext::afToHalideDims(dims), testing::ElementsAre(6, 3, 5, 1));

  Halide::Buffer<float> buffer(ext::afToHalideDims(dims));
  EXPECT_EQ(ext::halideToAfDims(buffer), dims);

  // AF doesn't support > 4 dimensions
  std::vector<int> wideDims = {1, 2, 3, 4, 5};
  Halide::Buffer<float> wideBuffer(wideDims);
  EXPECT_THROW(ext::halideToAfDims(wideBuffer), std::invalid_argument);

  // Zero dim
  std::vector<int> dimWithZero = {1, 2, 0, 3};
  Halide::Buffer<float> bufferWithZeroDim(dimWithZero);
  EXPECT_EQ(ext::halideToAfDims(bufferWithZeroDim), af::dim4(3, 1, 2, 1));
}

TEST(HalideTest, ConvertArray) {
  auto arr = af::randu({5, 4, 3, 2});
  auto arrCopy = arr.copy();
  {
    ext::HalideBufferWrapper<float> halideBufWrapper(arr);
    // Underlying memory should be the same
    DevicePtr arrPtr(arr);
    EXPECT_EQ(
        arrPtr.get(),
        reinterpret_cast<void*>(
            halideBufWrapper.getBuffer().raw_buffer()->device));
  }
  // The underlying Array should remain unchanged
  EXPECT_TRUE(fl::allClose(arr, arrCopy));
}

TEST(HalideTest, ConvertArrayManual) {
  auto arr = af::randu({5, 4, 3});
  auto halideBuffer = ext::detail::toHalideBuffer<float>(arr);

  const float* afHostPtr = arr.host<float>();

  halideBuffer.copy_to_host();
  const float* halideHostPtr = halideBuffer.data();
  for (size_t i = 0; i < arr.elements(); ++i) {
    EXPECT_FLOAT_EQ(halideHostPtr[i], afHostPtr[i]);
  }
}

TEST(HalideTest, RoundTripConvertArrayManual) {
  auto arr = af::randu({10, 14, 3});
  auto halideBuffer2 = ext::detail::toHalideBuffer<float>(arr);
  auto out = ext::detail::fromHalideBuffer(halideBuffer2);
  EXPECT_TRUE(fl::allClose(arr, out));
}

TEST(HalideTest, SimpleAOTCompiledHalidePipeline) {
  int yDim = 20, xDim = 20;
  auto inputVar = Variable((af::randu({yDim, xDim}) * 100), false);
  auto outputVar = Variable(af::array({yDim, xDim}, af::dtype::f32), false);

  // Halide buffers
  auto input = ext::HalideBufferWrapper<float>(inputVar.array());
  auto output = ext::HalideBufferWrapper<float>(outputVar.array());

  // This Halide AOT-generated pipeline pixel-wise adds an offset to each
  // element. It's has block and thread level tile-parallelism -- see the
  // accompanying schedule.
  int offset = 5;
  FL_HALIDE_CHECK(
      brighter(input.getRuntimeBuffer(), offset, output.getRuntimeBuffer()));

  cudaDeviceSynchronize();

  // Ensure the output buffer is correctly-modified
  EXPECT_TRUE(fl::allClose(inputVar.array() + offset, outputVar.array()));
}

TEST(HalideTest, SimpleJITHalidePipeline) {
  // Make sure we can call the Halide JIT inline in flashlight
  int yDim = 10, xDim = 10;

  Halide::Func sum("sum");
  Halide::Var x("x"), y("y");
  sum(x, y) = x + y;

  Halide::Buffer<int> out = sum.realize(xDim, yDim);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
