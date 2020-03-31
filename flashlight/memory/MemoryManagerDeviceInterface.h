/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>

namespace fl {

using GetActiveDeviceIdFn = std::function<int()>;
using GetMaxMemorySizeFn = std::function<size_t(int)>;
using NativeAllocFn = std::function<void*(size_t)>;
using NativeFreeFn = std::function<void(void*)>;
using GetMemoryPressureThresholdFn = std::function<float()>;
using SetMemoryPressureThresholdFn = std::function<void(float)>;

struct MemoryManagerDeviceInterface {
  // Native memory management functions
  GetActiveDeviceIdFn getActiveDeviceId;
  GetMaxMemorySizeFn getMaxMemorySize;
  NativeAllocFn nativeAlloc;
  NativeFreeFn nativeFree;
  // Memory pressure functions
  GetMemoryPressureThresholdFn getMemoryPressureThreshold;
  SetMemoryPressureThresholdFn setMemoryPressureThreshold;
};

} // namespace fl
