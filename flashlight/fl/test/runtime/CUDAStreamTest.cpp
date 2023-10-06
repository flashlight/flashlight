/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/tensor/Init.h"

#include <cuda_runtime.h>

using fl::DeviceManager;
using fl::DeviceType;
using fl::CUDAStream;
using fl::Stream;
using fl::StreamType;

TEST(CUDAStreamTest, createManaged) {
  auto& manager = DeviceManager::getInstance();
  for (auto cudaDevice : manager.getDevicesOfType(DeviceType::CUDA)) {
    cudaDevice->setActive();
    auto cudaStream = CUDAStream::createManaged();

    ASSERT_EQ(cudaStream->type, StreamType::CUDA);
    ASSERT_EQ(&cudaStream->device(), cudaDevice);
    ASSERT_EQ(&cudaStream->impl<CUDAStream>(), cudaStream.get());
  }
}

TEST(CUDAStreamTest, createUnmanaged) {
  auto& manager = DeviceManager::getInstance();
  for (auto cudaDevice : manager.getDevicesOfType(DeviceType::CUDA)) {
    cudaDevice->setActive();
    auto cudaStream = CUDAStream::createUnmanaged();

    ASSERT_EQ(cudaStream->type, StreamType::CUDA);
    ASSERT_EQ(&cudaStream->device(), cudaDevice);
    ASSERT_EQ(&cudaStream->impl<CUDAStream>(), cudaStream.get());
    // safe to destroy since underlying stream isn't managed.
    FL_CUDA_CHECK(cudaStreamDestroy(cudaStream->handle()));
  }
}


TEST(CUDAStreamTest, unmanagedWrapper) {
  auto& manager = DeviceManager::getInstance();
  int numCudaDevices = 0;
  FL_CUDA_CHECK(cudaGetDeviceCount(&numCudaDevices));

  for (int id = 0; id < numCudaDevices; id++) {
    FL_CUDA_CHECK(cudaSetDevice(id));
    cudaStream_t nativeStream;
    FL_CUDA_CHECK(cudaStreamCreate(&nativeStream));
    auto& cudaDevice = manager.getDevice(DeviceType::CUDA, id);
    auto cudaStream = CUDAStream::wrapUnmanaged(id, nativeStream);

    ASSERT_EQ(cudaStream->type, StreamType::CUDA);
    ASSERT_EQ(&cudaStream->device(), &cudaDevice);
    ASSERT_EQ(cudaStream->handle(), cudaStream->handle());
    ASSERT_EQ(&cudaStream->impl<CUDAStream>(), cudaStream.get());
    // safe to destroy since wrapper won't manage underlying stream by default.
    FL_CUDA_CHECK(cudaStreamDestroy(nativeStream));
  }
}

TEST(CUDAStreamTest, unmanagedWrapperDeviceSwitch) {
  auto& manager = DeviceManager::getInstance();
  if (manager.getDeviceCount(DeviceType::CUDA) > 1) {
    const auto& device0 = manager.getDevice(DeviceType::CUDA, 0);
    const auto& device1 = manager.getDevice(DeviceType::CUDA, 1);

    device0.setActive();
    cudaStream_t nativeStream;
    FL_CUDA_CHECK(cudaStreamCreate(&nativeStream));
    device1.setActive();
    // wrapper will switch to device0, create, and then switch back.
    auto cudaStreamWrapped = CUDAStream::wrapUnmanaged(device0.nativeId(), nativeStream);
    auto cudaStreamCreated = CUDAStream::createManaged();

    // nothing blows up -- event was created and used correctly
    ASSERT_NO_THROW(cudaStreamWrapped->relativeSync(*cudaStreamCreated));
    ASSERT_NO_THROW(cudaStreamCreated->relativeSync(*cudaStreamWrapped));

    ASSERT_EQ(&cudaStreamWrapped->device(), &device0);
    ASSERT_EQ(&cudaStreamCreated->device(), &device1);
    ASSERT_EQ(&manager.getActiveDevice(DeviceType::CUDA), &device1);
  }
}

TEST(CUDAStreamTest, relativeSync) {
  auto cs1 = CUDAStream::createManaged();
  auto cs2 = CUDAStream::createManaged();
  std::shared_ptr<Stream> s1 = cs1;
  std::shared_ptr<Stream> s2 = cs2;
  ASSERT_NO_THROW(s1->relativeSync(*s2));
  ASSERT_NO_THROW(s1->relativeSync(*cs2));
  ASSERT_NO_THROW(cs1->relativeSync(*s2));
  ASSERT_NO_THROW(cs1->relativeSync(*cs2));

  std::unordered_set<const Stream*> streams { s1.get(), s2.get() };
  std::shared_ptr<Stream> s3 = CUDAStream::createManaged();
  ASSERT_NO_THROW(s3->relativeSync(streams));
}

TEST(CUDAStreamTest, sync) {
  auto cs1 = CUDAStream::createManaged();
  ASSERT_NO_THROW(cs1->sync());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
