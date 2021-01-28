/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "flashlight/ext/integrations/halide/HalideInterface.h"
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"

// Generated at build time -- see the accompanying CMakeList
#include "flashlight/ext/test/HalideTestPipeline.h"

using namespace fl;

TEST(HalideTest, TypeMapping) {
  Halide::Buffer<Halide::float16_t> halfBuf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(halfBuf.type()), af::dtype::f16);

  Halide::Buffer<float> floatBuf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(floatBuf.type()), af::dtype::f32);

  Halide::Buffer<double> doubleBuf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(doubleBuf.type()), af::dtype::f64);

  Halide::Buffer<uint16_t> uint16Buf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(uint16Buf.type()), af::dtype::u16);

  Halide::Buffer<uint32_t> uint32Buf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(uint32Buf.type()), af::dtype::u32);

  Halide::Buffer<uint64_t> uint64Buf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(uint64Buf.type()), af::dtype::u64);

  Halide::Buffer<short> shortBuf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(shortBuf.type()), af::dtype::s16);

  Halide::Buffer<int> intBuf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(intBuf.type()), af::dtype::s32);

  Halide::Buffer<long> longBuf({1});
  EXPECT_EQ(ext::halideRuntimeTypeToAfType(longBuf.type()), af::dtype::s64);
}

TEST(HalideTest, ConvertDims) {
  af::dim4 emptyAfDims;
  EXPECT_EQ(ext::afToHalideDims(emptyAfDims).size(), 0);

  af::dim4 singleDims(3);
  EXPECT_THAT(ext::afToHalideDims(singleDims), testing::ElementsAre(3));

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
  EXPECT_EQ(ext::halideToAfDims(bufferWithZeroDim), af::dim4(0));

  std::vector<int> emptyDims = {};
  Halide::Buffer<float> bufferWithEmptyDim(emptyDims);
  EXPECT_EQ(ext::halideToAfDims(bufferWithEmptyDim), af::dim4(0));
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
  // The underlying Array should remain unchanged after the wrapper is destroyed
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
  int yDim = 240, xDim = 240;
  int offset = 5;

  auto input = af::randu({yDim, xDim}) * 100;
  auto expected = af::array(input.dims(), af::dtype::f32);
  auto output = af::array({yDim, xDim}, af::dtype::f32);

  // Reference implementation
  for (int i = 0; i < input.dims(0); ++i) {
    for (int j = 0; j < input.dims(1); ++j) {
      expected(i, j) = input(i, j) + std::sin(i * j) + offset;
    }
  }

  // Halide buffers
  auto inputHalide = ext::HalideBufferWrapper<float>(input);
  auto outputHalide = ext::HalideBufferWrapper<float>(output);

  // This Halide AOT-generated pipeline pixel-wise adds offsets to each
  // element. It's has block and thread level tile-parallelism -- see the
  // accompanying schedule.
  FL_HALIDE_CHECK(testFunc(
      inputHalide.getRuntimeBuffer(), offset, outputHalide.getRuntimeBuffer()));

  // Ensure the output buffer is correctly-modified
  EXPECT_TRUE(fl::allClose(expected, output));
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
