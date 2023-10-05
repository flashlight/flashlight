/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
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

/**
 * An interface for using native device memory management and JIT-related memory
 * pressure functions. Provides support for functions at the device and backend
 * level which automatically delegate to the correct backend functions for
 * native device interoperability. These functions call directly into ArrayFire
 * functions.
 *
 * Exposed as an external freestanding API so as to facilitate sharing native
 * device closures across different parts of a memory manager implemenation.
 *
 * Functions are automatically set when a `MemoryManagerDeviceInterface` that
 * has been passed to a constructed `MemoryManagerAdapter` is installed using a
 * `MemoryManagerInstaller`'s `setMemoryManager` or `setMemoryManagerPinned`
 * method. Until one of these are called, the functions therein remain unset.
 *
 * For documentation of virtual methods, see [ArrayFire's memory
 * header](https://git.io/Jv7do) for full specifications.
 */
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
