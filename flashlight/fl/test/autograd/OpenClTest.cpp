/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <iostream>
#include <stdexcept>

#include <CL/cl2.hpp>
#include <af/opencl.h>
#include <arrayfire.h>

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/OpenclUtils.h"
#include "flashlight/fl/common/common.h"

// based on code form
// https://gist.github.com/ShigekiKarita/edcab9d3797ff7633b73#file-kernel-cl-L14-L49

using namespace fl;

constexpr size_t kWIdx = 0;
constexpr size_t kHIdx = 1;
constexpr size_t kChannelSizeIdx = 2;
constexpr size_t kBatchSizeIdx = 3;

// const int ix = 4; // input.dims(kWIdx);
const int iy = 8; // input.dims(kHIdx);
// const int chan = 1;
// const int batch = 1;
const int wx = 2;
const int wy = 4;
const int sx = 2;
const int sy = 2;
// const int px = 0;
const int py = 2;

//   https://github.com/arrayfire/arrayfire/blob/096221d6e0040f092a259094d623e03856389ad5/src/backend/opencl/jit.cpp#L273

TEST(orig, mask) {
  for (int batch = 1; batch < 4; ++batch) {
    for (int chan = 1; chan < 4; ++chan) {
      for (int ix = 2; ix < 6; ix += 2) {
        for (int px = 0; px < 2; ++px) {
          const int ox = 1 + (ix + 2 * px - wx) / sx;
          const int oy = 1 + (iy + 2 * py - wy) / sy;

          auto input = Variable(af::randu(ix, iy, chan, batch), true);
          std::cout << "ix=" << ix << " iy=" << iy << " wx=" << wx
                    << " wy=" << wy << " sx=" << sx << " sy=" << sy
                    << " px=" << px << " py=" << py << " ox=" << ox
                    << " oy=" << oy << std::endl;
          af::print("input", input.array());

          auto output = pool2d(input, wx, wy, sx, sy, px, py, PoolingMode::MAX);
          // af::print("output", output.array());
          output.backward();
          af::print("output.grad", output.grad().array());
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
