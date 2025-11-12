/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireCPUStream.h"

using fl::DeviceManager;
using fl::DeviceType;
using fl::Stream;
using fl::StreamType;
using fl::ArrayFireCPUStream;




int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
