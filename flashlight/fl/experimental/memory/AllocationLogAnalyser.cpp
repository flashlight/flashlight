/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "flashlight/fl/common/Histogram.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/experimental/memory/AllocationLog.h"
#include "flashlight/fl/experimental/memory/allocator/ConfigurableMemoryAllocator.h"

using namespace fl;

constexpr size_t kArenaSize16gb = 16523001856; // (15GB+397MB+576KB)
constexpr size_t topSizeCountLen = 15;

int main(int argc, char** argv) {
  Logging::setMaxLoggingLevel(INFO);
  VerboseLogging::setMaxLoggingLevel(0);

  if (argc < 3) {
    FL_LOG(fl::ERROR)
        << "Usage:" << argv[0]
        << " [arenasize] [path to allocation log csv file] "
        << "\narenasize: use 16523001856 (15GB+397MB+576KB) for 16GB machines";
    return -1;
  }
  const size_t arenaSize = std::stol(argv[1]);
  if (arenaSize != kArenaSize16gb) {
    FL_LOG(fl::ERROR) << "Are you sure about the arena size? usueally it is:"
                      << kArenaSize16gb;
  }
  const std::string allocationLogPath = argv[2];

  std::ifstream allocationLogStream(allocationLogPath);
  if (!allocationLogStream.is_open()) {
    FL_LOG(fl::ERROR) << argv[0] << " failed to open allocation log file="
                      << allocationLogPath;
    return -1;
  }
  std::vector<AllocationEvent> allocationLog =
      LoadAllocationLog(allocationLogStream);
  if (allocationLog.empty()) {
    FL_LOG(fl::ERROR) << argv[0]
                      << " empty allocation log file=" << allocationLogPath;
    return -1;
  }
  FL_LOG(fl::INFO) << "Allocation log size="
                   << prettyStringCount(allocationLog.size());

  std::vector<size_t> allocationRequests;
  std::vector<size_t> freeRequests;
  std::vector<size_t> nativeAllocations;
  std::vector<size_t> cacheAllocations;
  std::vector<size_t> nativeFrees;
  std::vector<size_t> cacheFrees;
  size_t maxCurrentlyNativeAllocatedSum = 0;
  size_t maxCurrentlyNativeRequestedSum = 0;
  size_t currentlyNativeAllocatedSum = 0;
  size_t currentlyNativeRequestedSum = 0;
  size_t allNativeAllocatedSum = 0;
  size_t allNativeRequestedSum = 0;

  size_t maxCurrentlyCacheAllocatedSum = 0;
  size_t maxCurrentlyCacheRequestedSum = 0;
  size_t currentlyCacheAllocatedSum = 0;
  size_t currentlyCacheRequestedSum = 0;
  size_t allCacheAllocatedSum = 0;
  size_t allCacheRequestedSum = 0;

  std::unordered_map<void*, std::pair<size_t, size_t>>
      ptrToNativeRequestedAndAllocatedSizeMap;
  std::unordered_map<void*, std::pair<size_t, size_t>>
      ptrToCacheRequestedAndAllocatedSizeMap;
  std::unordered_map<size_t, size_t> allocationRequestSizeToCountMap;

  for (const AllocationEvent& event : allocationLog) {
    switch (event.type_) {
      case AllocationEvent::Type::ALLOCATE_NATIVE: {
        size_t& count = allocationRequestSizeToCountMap[event.sizeRequested_];
        ++count;
        nativeAllocations.push_back(event.sizeRequested_);
        allocationRequests.push_back(event.sizeRequested_);
        currentlyNativeRequestedSum += event.sizeRequested_;
        allNativeRequestedSum += event.sizeRequested_;
        currentlyNativeAllocatedSum += event.sizeAllocated_;
        allNativeAllocatedSum += event.sizeAllocated_;
        ptrToNativeRequestedAndAllocatedSizeMap[event.ptr_] = {
            event.sizeRequested_, event.sizeAllocated_};
        maxCurrentlyNativeAllocatedSum = std::max(
            maxCurrentlyNativeAllocatedSum, currentlyNativeAllocatedSum);
        maxCurrentlyNativeRequestedSum = std::max(
            maxCurrentlyNativeRequestedSum, currentlyNativeRequestedSum);
      } break;
      case AllocationEvent::Type::FREE_NATIVE: {
        auto ptrToSizeItr =
            ptrToNativeRequestedAndAllocatedSizeMap.find(event.ptr_);
        if (ptrToSizeItr == ptrToNativeRequestedAndAllocatedSizeMap.end()) {
          std::stringstream ss;
          ss << argv[0] << " attempts to free unalocated ptr=" << event.ptr_;
          FL_LOG(fl::WARNING) << ss.str();
          continue;
        }
        size_t sizeRequested = ptrToSizeItr->second.first;
        size_t sizeAllocated = ptrToSizeItr->second.second;
        currentlyNativeRequestedSum -= sizeRequested;
        currentlyNativeAllocatedSum -= sizeAllocated;
        freeRequests.push_back(sizeRequested);
        nativeFrees.push_back(sizeRequested);
      } break;
      case AllocationEvent::Type::ALLOCATE_CACHE: {
        size_t& count = allocationRequestSizeToCountMap[event.sizeRequested_];
        ++count;
        cacheAllocations.push_back(event.sizeRequested_);
        allocationRequests.push_back(event.sizeRequested_);
        currentlyCacheRequestedSum += event.sizeRequested_;
        allCacheRequestedSum += event.sizeRequested_;
        currentlyCacheAllocatedSum += event.sizeAllocated_;
        allCacheAllocatedSum += event.sizeAllocated_;
        ptrToCacheRequestedAndAllocatedSizeMap[event.ptr_] = {
            event.sizeRequested_, event.sizeAllocated_};
        maxCurrentlyCacheAllocatedSum =
            std::max(maxCurrentlyCacheAllocatedSum, currentlyCacheAllocatedSum);
        maxCurrentlyCacheRequestedSum =
            std::max(maxCurrentlyCacheRequestedSum, currentlyCacheRequestedSum);
      } break;
      case AllocationEvent::Type::FREE_CACHE: {
        auto ptrToSizeItr =
            ptrToCacheRequestedAndAllocatedSizeMap.find(event.ptr_);
        if (ptrToSizeItr == ptrToCacheRequestedAndAllocatedSizeMap.end()) {
          // std::stringstream ss;
          // ss << argv[0] << " attempts to free unalocated ptr=" << event.ptr_;
          // FL_LOG(fl::WARNING) << ss.str();
          continue;
        }
        size_t sizeRequested = ptrToSizeItr->second.first;
        size_t sizeAllocated = ptrToSizeItr->second.second;
        currentlyCacheRequestedSum -= sizeRequested;
        currentlyCacheAllocatedSum -= sizeAllocated;
        freeRequests.push_back(sizeRequested);
        cacheFrees.push_back(sizeRequested);
      } break;
      default: {
        FL_LOG(fl::ERROR) << "Invalid event.type_="
                          << static_cast<int>(event.type_);
      } break;
    }
  }

  const double cacheInternalFragmentation =
      ((1.0 -
        (static_cast<double>(allCacheRequestedSum) /
         static_cast<double>(allCacheAllocatedSum))) *
       100.0);

  HistogramStats<size_t> allocationRequestsHist =
      FixedBucketSizeHistogram<size_t>(
          allocationRequests.begin(),
          allocationRequests.end(),
          kHistogramBucketCountPrettyString);
  HistogramStats<size_t> freeRequestsHist = FixedBucketSizeHistogram<size_t>(
      freeRequests.begin(),
      freeRequests.end(),
      kHistogramBucketCountPrettyString);
  HistogramStats<size_t> nativeAllocationsHist =
      FixedBucketSizeHistogram<size_t>(
          nativeAllocations.begin(),
          nativeAllocations.end(),
          kHistogramBucketCountPrettyString);
  HistogramStats<size_t> nativeFreesHist = FixedBucketSizeHistogram<size_t>(
      nativeFrees.begin(),
      nativeFrees.end(),
      kHistogramBucketCountPrettyString);
  HistogramStats<size_t> cacheAllocationsHist =
      FixedBucketSizeHistogram<size_t>(
          cacheAllocations.begin(),
          cacheAllocations.end(),
          kHistogramBucketCountPrettyString);
  HistogramStats<size_t> cacheFreesHist = FixedBucketSizeHistogram<size_t>(
      cacheFrees.begin(), cacheFrees.end(), kHistogramBucketCountPrettyString);

  std::stringstream ss;
  ss << std::endl
     << "nativeInternalFragmentation="
     << ((1.0 -
          (static_cast<double>(allNativeRequestedSum) /
           static_cast<double>(allNativeAllocatedSum))) *
         100)
     << '%' << " allNativeRequestedSum="
     << prettyStringMemorySize(allNativeRequestedSum)
     << " allNativeAllocatedSum="
     << prettyStringMemorySize(allNativeAllocatedSum)
     << " (allNativeAllocatedSum-allNativeRequestedSum)="
     << prettyStringMemorySize(allNativeAllocatedSum - allNativeRequestedSum)
     << " (allNativeRequestedSum/allNativeAllocatedSum)="
     << (static_cast<double>(allNativeRequestedSum) /
         static_cast<double>(allNativeAllocatedSum))
     << std::endl;
  ss << "cacheInternalFragmentation=" << cacheInternalFragmentation << '%'
     << " allNativeRequestedSum="
     << prettyStringMemorySize(allCacheRequestedSum)
     << " allCacheAllocatedSum=" << prettyStringMemorySize(allCacheAllocatedSum)
     << " (allCacheAllocatedSum-allCacheRequestedSum)="
     << prettyStringMemorySize(allCacheAllocatedSum - allCacheRequestedSum)
     << " (allCacheRequestedSum/allCacheAllocatedSum)="
     << (static_cast<double>(allCacheRequestedSum) /
         static_cast<double>(allCacheAllocatedSum))
     << std::endl;
  ss << "maxCurrentlyNativeAllocatedSum="
     << prettyStringMemorySize(maxCurrentlyNativeAllocatedSum) << std::endl
     << "maxCurrentlyNativeRequestedSum="
     << prettyStringMemorySize(maxCurrentlyNativeRequestedSum) << std::endl
     << "maxCurrentlyNativeAllocatedSum-maxCurrentlyNativeRequestedSum="
     << prettyStringMemorySize(
            maxCurrentlyCacheAllocatedSum - maxCurrentlyCacheRequestedSum)
     << std::endl;
  ss << "maxCurrentlyCacheAllocatedSum="
     << prettyStringMemorySize(maxCurrentlyCacheAllocatedSum) << std::endl
     << "maxCurrentlyCacheRequestedSum="
     << prettyStringMemorySize(maxCurrentlyCacheRequestedSum) << std::endl
     << "maxCurrentlyCacheAllocatedSum-maxCurrentlyCacheRequestedSum="
     << prettyStringMemorySize(
            maxCurrentlyCacheAllocatedSum - maxCurrentlyCacheRequestedSum)
     << std::endl
     << std::endl;
  ss << "All allocation requests histogram:" << std::endl
     << allocationRequestsHist.prettyString() << std::endl
     << std::endl
     << "All free requests histogram:" << std::endl
     << freeRequestsHist.prettyString() << std::endl
     << std::endl
     << "Native allocation requests histogram:" << std::endl
     << nativeAllocationsHist.prettyString() << std::endl
     << std::endl
     << "Native free requests histogram:" << std::endl
     << nativeFreesHist.prettyString() << std::endl
     << std::endl
     << "Cache allocation requests histogram:" << std::endl
     << cacheAllocationsHist.prettyString() << std::endl
     << std::endl
     << "Cache free requests histogram:" << std::endl
     << cacheFreesHist.prettyString() << std::endl
     << std::endl;

  std::vector<size_t> allocationRequestSize;
  std::vector<size_t> allocationRequestSizeCount;
  std::vector<size_t> allocationRequestSizeIndex(
      allocationRequestSizeToCountMap.size());
  std::iota(
      allocationRequestSizeIndex.begin(), allocationRequestSizeIndex.end(), 0);

  for (const auto& sizeToCnt : allocationRequestSizeToCountMap) {
    allocationRequestSize.push_back(sizeToCnt.first);
    allocationRequestSizeCount.push_back(sizeToCnt.second);
  }
  std::sort(
      allocationRequestSizeIndex.begin(),
      allocationRequestSizeIndex.end(),
      [&allocationRequestSizeCount](int lhs, int rhs) {
        return allocationRequestSizeCount[lhs] >
            allocationRequestSizeCount[rhs];
      });
  const double numAllocationRequests = allocationRequests.size();
  ss << "# rank," << std::setw(30) << "allocationRequestSize," << std::setw(30)
     << "count,"
     << " %of count, total bytes alocated" << std::endl;
  for (int i = 0; i < topSizeCountLen; ++i) {
    const int idx = allocationRequestSizeIndex.at(i);
    ss << std::setw(2) << i << ", " << std::setw(30)
       << prettyStringMemorySize(allocationRequestSize.at(idx)) << ", "
       << std::setw(30) << prettyStringCount(allocationRequestSizeCount.at(idx))
       << ", "
       << ((static_cast<double>(allocationRequestSizeCount.at(idx)) /
            numAllocationRequests) *
           100.0)
       << ", "
       << prettyStringMemorySize(
              allocationRequestSize.at(idx) *
              allocationRequestSizeCount.at(idx))
       << std::endl;
  }
  FL_LOG(fl::INFO) << ss.str();
  return 0;
}
