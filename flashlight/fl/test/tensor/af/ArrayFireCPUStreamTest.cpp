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

TEST(ArrayFireCPUStreamTest, create) {
  const auto& manager = DeviceManager::getInstance();
  for (const auto x64Device : manager.getDevicesOfType(DeviceType::x64)) {
    x64Device->setActive();
    const auto afCpuStream = ArrayFireCPUStream::create();

    ASSERT_EQ(afCpuStream->type, StreamType::Synchronous);
    ASSERT_EQ(&afCpuStream->device(), x64Device);
    ASSERT_EQ(&afCpuStream->impl<ArrayFireCPUStream>(), afCpuStream.get());
  }
}

TEST(ArrayFireCPUStreamTest, relativeSync) {
  const auto as1 = ArrayFireCPUStream::create();
  const auto as2 = ArrayFireCPUStream::create();
  const std::shared_ptr<Stream> s1 = as1;
  const std::shared_ptr<Stream> s2 = as2;
  ASSERT_NO_THROW(s1->relativeSync(*s2));
  ASSERT_NO_THROW(s1->relativeSync(*as2));
  ASSERT_NO_THROW(as1->relativeSync(*s2));
  ASSERT_NO_THROW(as1->relativeSync(*as2));

  const std::unordered_set<const Stream*> streams { s1.get(), s2.get() };
  const std::shared_ptr<Stream> s3 = ArrayFireCPUStream::create();
  ASSERT_NO_THROW(s3->relativeSync(streams));
}

TEST(ArrayFireCPUStreamTest, sync) {
  const auto as1 = ArrayFireCPUStream::create();
  ASSERT_NO_THROW(as1->sync());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
