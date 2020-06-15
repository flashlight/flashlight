/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <ostream>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <functional>

#include "flashlight/experimental/memory/allocator/MemoryAllocator.h"
#include "flashlight/experimental/memory/AllocationLog.h"

namespace fl {

// Replays the allocation log over the given allocator. After calling this
// function the allocator's stats indicate how well the allocator is configured
// for the given allocation. Returns false on OOM.
// Debug info is written to os when not null.
bool simulateAllocatorOnAllocationLog(
    const std::vector<AllocationEvent>& allocationLog,
    MemoryAllocator* allocator);

class BlockingThreadPool {
 public:
  explicit BlockingThreadPool(size_t nThreads);

  void enqueue(std::function<void()> task);

  ~BlockingThreadPool();

  void blockUntilAlldone();

 private:
  void workerFunction();

  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  // synchronization
  std::mutex mutex_;
  std::condition_variable workerWakeUpCondition_;
  std::condition_variable allDoneCondition_;
  int currentlyRunningCount_;

  bool stop_;
};

struct SimResult {
  SimResult();
  std::string prettyString() const;

  bool success_;
  double timeElapsedNanoSec_;
};

std::vector<SimResult> simulateAllocatorsOnAllocationLog(
    BlockingThreadPool& threadPool,
    const std::vector<AllocationEvent>& allocationLog,
    std::vector<MemoryAllocator*>& allocators);

} // namespace fl
