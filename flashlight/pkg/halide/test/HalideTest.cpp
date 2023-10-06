/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/pkg/halide/HalideInterface.h"

// Generated at build time -- see the accompanying CMakeList
#include "HalideTestPipeline.h"

using namespace fl;

TEST(HalideTest, TypeMapping) {
  Halide::Buffer<Halide::float16_t> halfBuf({1});
  EXPECT_EQ(
      pkg::halide::halideRuntimeTypeToFlType(halfBuf.type()), fl::dtype::f16);

  Halide::Buffer<float> floatBuf({1});
  EXPECT_EQ(
      pkg::halide::halideRuntimeTypeToFlType(floatBuf.type()), fl::dtype::f32);

  Halide::Buffer<double> doubleBuf({1});
  EXPECT_EQ(
      pkg::halide::halideRuntimeTypeToFlType(doubleBuf.type()), fl::dtype::f64);

  Halide::Buffer<uint16_t> uint16Buf({1});
  EXPECT_EQ(
      pkg::halide::halideRuntimeTypeToFlType(uint16Buf.type()), fl::dtype::u16);

  Halide::Buffer<uint32_t> uint32Buf({1});
  EXPECT_EQ(
      pkg::halide::halideRuntimeTypeToFlType(uint32Buf.type()), fl::dtype::u32);

  Halide::Buffer<uint64_t> uint64Buf({1});
  EXPECT_EQ(
      pkg::halide::halideRuntimeTypeToFlType(uint64Buf.type()), fl::dtype::u64);

  Halide::Buffer<short> shortBuf({1});
  EXPECT_EQ(
      pkg::halide::halideRuntimeTypeToFlType(shortBuf.type()), fl::dtype::s16);

  Halide::Buffer<int> intBuf({1});
  EXPECT_EQ(
      pkg::halide::halideRuntimeTypeToFlType(intBuf.type()), fl::dtype::s32);

  Halide::Buffer<long> longBuf({1});
  EXPECT_EQ(
      pkg::halide::halideRuntimeTypeToFlType(longBuf.type()), fl::dtype::s64);
}

TEST(HalideTest, ConvertDims) {
  Shape emptyFlDims;
  EXPECT_EQ(pkg::halide::flToHalideDims(emptyFlDims).size(), 0);

  Shape singleDims({3});
  EXPECT_THAT(pkg::halide::flToHalideDims(singleDims), testing::ElementsAre(3));

  Shape dims({1, 5, 3, 6});
  EXPECT_THAT(
      pkg::halide::flToHalideDims(dims), testing::ElementsAre(6, 3, 5, 1));

  Halide::Buffer<float> buffer(pkg::halide::flToHalideDims(dims));
  EXPECT_EQ(pkg::halide::halideToFlDims(buffer), dims);

  // Zero dim
  std::vector<int> dimWithZero = {1, 2, 0, 3};
  Halide::Buffer<float> bufferWithZeroDim(dimWithZero);
  EXPECT_EQ(pkg::halide::halideToFlDims(bufferWithZeroDim), Shape({0}));

  std::vector<int> emptyDims = {};
  Halide::Buffer<float> bufferWithEmptyDim(emptyDims);
  EXPECT_EQ(pkg::halide::halideToFlDims(bufferWithEmptyDim), Shape());
}

TEST(HalideTest, ConvertArray) {
  auto arr = fl::rand({5, 4, 3, 2});
  auto arrCopy = arr.copy();
  {
    pkg::halide::HalideBufferWrapper<float> halideBufWrapper(arr);
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
  auto arr = fl::rand({5, 4, 3});
  auto halideBuffer = pkg::halide::detail::toHalideBuffer<float>(arr);

  const float* flHostPtr = arr.host<float>();

  halideBuffer.copy_to_host();
  const float* halideHostPtr = halideBuffer.data();
  for (size_t i = 0; i < arr.elements(); ++i) {
    EXPECT_FLOAT_EQ(halideHostPtr[i], flHostPtr[i]);
  }
}

TEST(HalideTest, RoundTripConvertArrayManual) {
  auto arr = fl::rand({10, 14, 3});
  auto halideBuffer2 = pkg::halide::detail::toHalideBuffer<float>(arr);
  auto out = pkg::halide::detail::fromHalideBuffer(halideBuffer2);
  EXPECT_TRUE(fl::allClose(arr, out));
}

TEST(HalideTest, SimpleAOTCompiledHalidePipeline) {
  int yDim = 240, xDim = 240;
  int offset = 5;

  auto input = fl::rand({yDim, xDim}) * 100;
  auto expected = Tensor(input.shape(), fl::dtype::f32);
  auto output = Tensor({yDim, xDim}, fl::dtype::f32);

  // Reference implementation
  for (int i = 0; i < input.dim(0); ++i) {
    for (int j = 0; j < input.dim(1); ++j) {
      expected(i, j) = input(i, j) + std::sin(i * j) + offset;
    }
  }

  // Halide buffers
  auto inputHalide = pkg::halide::HalideBufferWrapper<float>(input);
  auto outputHalide = pkg::halide::HalideBufferWrapper<float>(output);

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

  Halide::Buffer<int> out = sum.realize({xDim, yDim});
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
