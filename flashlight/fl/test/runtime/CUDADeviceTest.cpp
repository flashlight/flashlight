/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/runtime/CUDADevice.h"
#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/Init.h"

#include <cuda_runtime.h>

using fl::CUDADevice;
using fl::DeviceManager;
using fl::DeviceType;

TEST(CUDADeviceTest, impl) {
  auto& manager = DeviceManager::getInstance();

  auto& cudaDevice = manager.getActiveDevice(DeviceType::CUDA);
  ASSERT_NO_THROW(cudaDevice.impl<fl::CUDADevice>());
  ASSERT_THROW(cudaDevice.impl<fl::X64Device>(), std::invalid_argument);

  auto& x64Device = manager.getActiveDevice(DeviceType::x64);
  ASSERT_NO_THROW(x64Device.impl<fl::X64Device>());
  ASSERT_THROW(x64Device.impl<fl::CUDADevice>(), std::invalid_argument);
}

TEST(CUDADeviceTest, nativeId) {
  auto& manager = DeviceManager::getInstance();
  int numCudaDevices = 0;
  cudaGetDeviceCount(&numCudaDevices);

  for (auto id = 0; id < numCudaDevices; id++) {
    auto& cudaDevice =
      manager.getDevice(DeviceType::CUDA, id).impl<CUDADevice>();
    ASSERT_EQ(cudaDevice.nativeId(), id);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
