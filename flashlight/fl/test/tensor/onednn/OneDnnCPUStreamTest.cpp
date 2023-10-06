/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/backend/onednn/OneDnnCPUStream.h"

#include <dnnl.hpp>

using fl::DeviceManager;
using fl::DeviceType;
using fl::Stream;
using fl::StreamType;
using fl::OneDnnCPUStream;

TEST(OneDnnCPUStreamTest, create) {
  const dnnl::engine cpuEngine(dnnl::engine::kind::cpu, 0);
  const auto& manager = DeviceManager::getInstance();
  const auto& x64Device = manager.getActiveDevice(DeviceType::x64);
  const auto stream = OneDnnCPUStream::create(cpuEngine);

  ASSERT_EQ(stream->type, StreamType::Synchronous);
  ASSERT_EQ(&stream->device(), &x64Device);
  ASSERT_EQ(&stream->impl<OneDnnCPUStream>(), stream.get());
}

TEST(OneDnnCPUStreamTest, relativeSync) {
  const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  const auto os1 = OneDnnCPUStream::create(engine);
  const auto os2 = OneDnnCPUStream::create(engine);
  const std::shared_ptr<Stream> s1 = os1;
  const std::shared_ptr<Stream> s2 = os2;
  ASSERT_NO_THROW(s1->relativeSync(*s2));
  ASSERT_NO_THROW(s1->relativeSync(*os2));
  ASSERT_NO_THROW(os1->relativeSync(*s2));
  ASSERT_NO_THROW(os1->relativeSync(*os2));

  const std::unordered_set<const Stream*> streams { s1.get(), s2.get() };
  const std::shared_ptr<Stream> s3 = OneDnnCPUStream::create(engine);
  ASSERT_NO_THROW(s3->relativeSync(streams));
}

TEST(OneDnnCPUStreamTest, sync) {
  const dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  const auto os1 = OneDnnCPUStream::create(engine);
  ASSERT_NO_THROW(os1->sync());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
