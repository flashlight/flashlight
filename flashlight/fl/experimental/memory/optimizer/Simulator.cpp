/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/experimental/memory/optimizer/Simulator.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/experimental/memory/AllocationLog.h"

namespace fl {

bool simulateAllocatorOnAllocationLog(
    const std::vector<AllocationEvent>& allocationLog,
    MemoryAllocator* allocator) {
  std::unordered_map<void*, void*> logPtrToAllocatorPtr;

  for (const AllocationEvent& event : allocationLog) {
    switch (event.type_) {
      case AllocationEvent::Type::ALLOCATE_NATIVE: {
        void* allocatorPtr = nullptr;
        try {
          allocatorPtr = allocator->allocate(event.sizeRequested_);
        } catch (std::exception& ex) {
          FL_LOG(fl::ERROR)
              << "simulateAllocatorOnAllocationLog() allocator->allocate(event.sizeRequested_="
              << event.sizeRequested_
              << " OOM. allocator=" << allocator->getName();
          return false;
        }
        if (!allocatorPtr) {
          FL_LOG(fl::ERROR)
              << "simulateAllocatorOnAllocationLog() allocator->allocate(event.sizeRequested_="
              << event.sizeRequested_ << ") returns null."
              << " allocator=" << allocator->getName();
          return false;
        }
        logPtrToAllocatorPtr[event.ptr_] = allocatorPtr;
      } break;
      case AllocationEvent::Type::FREE_NATIVE: {
        auto logPtrItr = logPtrToAllocatorPtr.find(event.ptr_);
        if (logPtrItr == logPtrToAllocatorPtr.end()) {
          // std::stringstream ss;
          // ss << "simulateAllocatorOnAllocationLog() attempts to free
          // unalocated ptr="
          //    << event.ptr_ << std::endl;
          // FL_LOG(fl::ERROR) << ss.str();
          // throw std::invalid_argument(ss.str());
          // return false;
          continue;
        }
        void* allocatorPtr = logPtrItr->second;
        size_t size = event.sizeRequested_;
        if (size == 0) {
          size = allocator->getAllocatedSizeInBytes(allocatorPtr);
        }
        try {
          allocator->free(allocatorPtr);
        } catch (std::exception& ex) {
          FL_LOG(fl::ERROR)
              << "simulateAllocatorOnAllocationLog() allocator->free(allocatorPtr="
              << allocatorPtr << " error=" << ex.what()
              << ". allocator=" << allocator->getName();
          continue;
        }
      } break;
      default: {
        FL_LOG(fl::ERROR)
            << "simulateAllocatorOnAllocationLog() invalid event.type_="
            << static_cast<int>(event.type_);
      } break;
    }
  }

  return true;
}

long timespec_diff_nsec(struct timespec end, struct timespec start) {
  long nStartSec = start.tv_nsec + start.tv_sec * 1e9;
  long nEndSec = end.tv_nsec + end.tv_sec * 1e9;
  return nEndSec - nStartSec;
}

SimResult::SimResult() : success_(false), timeElapsedNanoSec_(0.0) {}

std::string SimResult::prettyString() const {
  std::stringstream ss;
  ss << "success_=" << success_
     << " timeElapsedNanoSec_=" << timeElapsedNanoSec_;
  return ss.str();
}

std::vector<SimResult> simulateAllocatorsOnAllocationLog(
    BlockingThreadPool& threadPool,
    const std::vector<AllocationEvent>& allocationLog,
    std::vector<MemoryAllocator*>& allocators) {
  std::stringstream ss;
  ss << "simulateAllocatorsOnAllocationLog(threadPool,allocationLog.size()="
     << allocationLog.size() << ", allocators.size()=" << allocators.size()
     << ")";

  if (allocators.empty()) {
    ss << " empty allocators vector";
    throw std::invalid_argument(ss.str());
  }
  if (allocationLog.empty()) {
    ss << " empty allocation log";
    throw std::invalid_argument(ss.str());
  }

  std::vector<SimResult> results(allocators.size());
  try {
    for (int i = 0; i < allocators.size(); ++i) {
      MemoryAllocator* allocator = allocators[i];
      threadPool.enqueue([&ss, allocationLog, allocator, &results, i]() {
        struct timespec startTime;
        struct timespec endTime;
        clockid_t threadClockId;
        //! Get thread clock Id
        pthread_getcpuclockid(pthread_self(), &threadClockId);
        //! Using thread clock Id get the clock time
        clock_gettime(threadClockId, &startTime);

        SimResult& simResult = results[i];
        try {
          simResult.success_ =
              simulateAllocatorOnAllocationLog(allocationLog, allocator);
          clock_gettime(threadClockId, &endTime);
          simResult.timeElapsedNanoSec_ =
              timespec_diff_nsec(endTime, startTime);
          FL_LOG(fl::ERROR)
              << "Allocator=" << allocator->getName()
              << " simResult=" << simResult.prettyString() << " i=" << i;
        } catch (std::exception& ex) {
          FL_LOG(fl::ERROR) << ss.str() << " threadPool.enqueue(i=" << i
                            << ") failed with error=" << ex.what();
        }
      });
    }
  } catch (std::exception& e) {
    FL_LOG(fl::ERROR) << e.what();
  }
  threadPool.blockUntilAlldone();
  return results;
}

void BlockingThreadPool::blockUntilAlldone() {
  std::unique_lock<std::mutex> lock(mutex_);
  this->allDoneCondition_.wait(lock, [this] {
    const bool allDone =
        this->tasks_.empty() && this->currentlyRunningCount_ < 1;
    FL_LOG(fl::INFO) << " allDone=" << allDone;
    return allDone;
  });
}

BlockingThreadPool::BlockingThreadPool(size_t nThreads)
    : currentlyRunningCount_(0), stop_(false) {
  for (size_t id = 0; id < nThreads; ++id) {
    // std::thread t(&BlockingThreadPool::workerFunction, this);
    workers_.push_back(std::thread(&BlockingThreadPool::workerFunction, this));
  }
}

BlockingThreadPool::~BlockingThreadPool() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!tasks_.empty() || currentlyRunningCount_ > 0) {
      FL_LOG(fl::WARNING)
          << "BlockingThreadPool::~BlockingThreadPool() tasks_.size()="
          << tasks_.size()
          << " currentlyRunningCount_=" << currentlyRunningCount_;
    }
    stop_ = true;
  }
  workerWakeUpCondition_.notify_all();
  for (std::thread& worker : workers_) {
    worker.join();
  }
}

void BlockingThreadPool::enqueue(std::function<void()> task) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (stop_) {
    throw std::runtime_error("enqueue on stopped BlockingThreadPool");
  }

  tasks_.push(std::move(task));
  workerWakeUpCondition_.notify_one();
}

void BlockingThreadPool::workerFunction() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      workerWakeUpCondition_.wait(
          lock, [this] { return this->stop_ || !this->tasks_.empty(); });

      if (stop_) {
        return;
      }

      if (tasks_.empty()) {
        FL_LOG(fl::ERROR)
            << "BlockingThreadPool::workerFunction() tasks_.empty() where it should never be the case.";
      } else {
        ++currentlyRunningCount_;
        task = std::move(this->tasks_.front());
        this->tasks_.pop();
      }
    }

    if (task) {
      try {
        task();
      } catch (std::exception& ex) {
        FL_LOG(fl::ERROR) << "BlockingThreadPool::workerFunction() exception="
                          << ex.what();
      }
      std::unique_lock<std::mutex> lock(mutex_);
      if (currentlyRunningCount_ < 1) {
        FL_LOG(fl::ERROR)
            << "BlockingThreadPool::workerFunction() currentlyRunningCount_ goes negative";
      } else {
        --currentlyRunningCount_;
      }
      const bool allDone =
          this->tasks_.empty() && this->currentlyRunningCount_ < 1;
      if (allDone) {
        FL_LOG(fl::INFO)
            << "BlockingThreadPool::workerFunction() allDoneCondition_ notify_all";
        allDoneCondition_.notify_all();
      }
    } else {
      FL_LOG(fl::ERROR)
          << "BlockingThreadPool::workerFunction() tasks is null where it should never be the case.";
    }
  }
}

}; // namespace fl
