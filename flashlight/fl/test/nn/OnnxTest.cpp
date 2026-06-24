/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/nn/onnx/Onnx.h"
#include "flashlight/fl/tensor/Init.h"

using namespace fl;

TEST(OnnxTest, LoadModel) {
  const std::string path = "/Users/jacobkahn/Downloads/bvlcalexnet-3.onnx";

  auto foo = loadOnnxModuleFromPath(path);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
