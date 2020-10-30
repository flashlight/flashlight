/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/experimental/memory/allocator/ConfigurableMemoryAllocator.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <cereal/archives/json.hpp>

#include "flashlight/fl/common/CppBackports.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/experimental/memory/allocator/CompositeMemoryAllocator.h"
#include "flashlight/fl/experimental/memory/allocator/freelist/FreeList.h"
#include "flashlight/fl/experimental/memory/allocator/memorypool/MemoryPool.h"

namespace fl {

namespace {

bool almostEqual(double lhs, double rhs) {
  return std::fabs(lhs - rhs) < std::numeric_limits<double>::epsilon();
}

} // namespace

std::unique_ptr<MemoryAllocator> CreateMemoryAllocator(
    MemoryAllocatorConfiguration config,
    void* arenaAddress,
    size_t arenaSizeInBytes,
    int logLevel) {
  std::stringstream ss;
  ss << "CreateMemoryAllocator(arenaAddress=" << arenaAddress
     << " arenaSizeInBytes=" << prettyStringMemorySize(arenaSizeInBytes)
     << ") config=" << config.prettyString();
  if (logLevel > 0) {
    FL_LOG(fl::INFO) << ss.str();
  }

  const size_t nSubArenas = config.subArenaConfiguration_.size();
  if (nSubArenas == 0) {
    ss << " config must have at least one sub arena configuration.";
    FL_LOG(fl::ERROR) << ss.str();
    throw std::invalid_argument(ss.str());
  }

  config.normalize();

  // Sort by sub arena relative size from big to small.
  std::vector<size_t> subArenaReverseRelativeSizeSorting(nSubArenas);
  std::iota(
      subArenaReverseRelativeSizeSorting.begin(),
      subArenaReverseRelativeSizeSorting.end(),
      0);
  std::sort(
      subArenaReverseRelativeSizeSorting.begin(),
      subArenaReverseRelativeSizeSorting.end(),
      [&config](size_t lhs, size_t rhs) {
        return config.subArenaConfiguration_[lhs].relativeSize_ >
            config.subArenaConfiguration_[rhs].relativeSize_;
      });

  auto compositeAllocator =
      fl::cpp::make_unique<CompositeMemoryAllocator>(config.name_);

  size_t avaialableBytes = arenaSizeInBytes;
  for (size_t i : subArenaReverseRelativeSizeSorting) {
    const size_t usedBytes = arenaSizeInBytes - avaialableBytes;
    void* subArenaAddress = static_cast<char*>(arenaAddress) + usedBytes;
    const SubArenaConfiguration& subArenaConfig =
        config.subArenaConfiguration_[i];
    const size_t subAreanRatioSize = static_cast<size_t>(
        static_cast<double>(arenaSizeInBytes) * subArenaConfig.relativeSize_);
    const size_t subArenaSize = std::min(subAreanRatioSize, avaialableBytes);
    const size_t subAreanClumpedSize =
        ((subArenaSize / subArenaConfig.blockSize_) *
         subArenaConfig.blockSize_);
    avaialableBytes -= subAreanClumpedSize;

    // FL_LOG(fl::INFO) << "CreateMemoryAllocator() arenaSizeInBytes="
    //           << prettyStringMemorySize(arenaSizeInBytes)
    //           << " subArenaConfig.relativeSize_="
    //           << subArenaConfig.relativeSize_
    //           << " (arenaSizeInBytes*subArenaConfig.relativeSize_)="
    //           << prettyStringMemorySize(subAreanRatioSize)
    //           << " subAreanClumpedSize="
    //           << prettyStringMemorySize(subAreanClumpedSize)
    //           << " avaialableBytes=" <<
    //           prettyStringMemorySize(avaialableBytes)
    //           << " usedBytes=" << prettyStringMemorySize(usedBytes)
    //           << " subArenaAddress=" << subArenaAddress
    //           << " config.subArenaConfiguration_[i=" << i
    //           << "]=" << subArenaConfig.prettyString();

    std::unique_ptr<MemoryAllocator> subAllocator;
    if (subArenaConfig.blockSize_ < subArenaConfig.maxAllocationSize_) {
      subAllocator = fl::cpp::make_unique<FreeList>(
          subArenaConfig.name_,
          subArenaAddress,
          subAreanClumpedSize,
          subArenaConfig.blockSize_,
          subArenaConfig.allocatedRatioJitThreshold_,
          logLevel);
    } else {
      subAllocator = fl::cpp::make_unique<MemoryPool>(
          subArenaConfig.name_,
          subArenaAddress,
          subAreanClumpedSize,
          subArenaConfig.blockSize_,
          subArenaConfig.allocatedRatioJitThreshold_,
          logLevel);
    }
    compositeAllocator->add(
        {subArenaConfig.maxAllocationSize_, std::move(subAllocator)});
  }
  compositeAllocator->setLogLevel(logLevel);
  return std::move(compositeAllocator);
}

MemoryAllocatorConfiguration::MemoryAllocatorConfiguration(
    std::string name,
    size_t alignmentNumberOfBits,
    std::vector<SubArenaConfiguration> subArenaConfiguration)
    : name_(std::move(name)),
      alignmentNumberOfBits_(alignmentNumberOfBits),
      subArenaConfiguration_(subArenaConfiguration) {}

MemoryAllocatorConfiguration::MemoryAllocatorConfiguration()
    : alignmentNumberOfBits_(0) {}

MemoryAllocatorConfiguration MemoryAllocatorConfiguration::loadJSon(
    std::istream& streamToConfig) {
  MemoryAllocatorConfiguration config;
  try {
    cereal::JSONInputArchive archive(streamToConfig);
    archive(config);
  } catch (std::exception& ex) {
    FL_LOG(fl::ERROR)
        << "MemoryAllocatorConfiguration::loadJSon() failed to load config with error="
        << ex.what();
    throw ex;
  }
  return config;
}

void MemoryAllocatorConfiguration::saveJSon(
    std::ostream& saveConfigStream) const {
  try {
    cereal::JSONOutputArchive archive(saveConfigStream);
    archive(*this);
  } catch (std::exception& ex) {
    FL_LOG(fl::ERROR)
        << "MemoryAllocatorConfiguration::saveJSon() failed to save config with error="
        << ex.what();
    throw ex;
  }
}

SubArenaConfiguration::SubArenaConfiguration()
    : blockSize_(0), maxAllocationSize_(0), relativeSize_(0.0) {}

bool SubArenaConfiguration::operator<(
    const SubArenaConfiguration& other) const {
  return blockSize_ < other.blockSize_ ||
      maxAllocationSize_ < other.maxAllocationSize_ ||
      relativeSize_ < other.relativeSize_;
}

bool SubArenaConfiguration::operator==(
    const SubArenaConfiguration& other) const {
  return blockSize_ == other.blockSize_ &&
      maxAllocationSize_ == other.maxAllocationSize_ &&
      almostEqual(relativeSize_, other.relativeSize_);
}

bool SubArenaConfiguration::operator!=(
    const SubArenaConfiguration& other) const {
  return !(*this == other);
}

SubArenaConfiguration::SubArenaConfiguration(
    std::string name,
    size_t blockSize,
    size_t maxAllocationSize,
    double relativeSize,
    double allocatedRatioJitThreshold)
    : name_(name),
      blockSize_(blockSize),
      maxAllocationSize_(maxAllocationSize),
      relativeSize_(relativeSize),
      allocatedRatioJitThreshold_(allocatedRatioJitThreshold) {}

std::string SubArenaConfiguration::prettyString() const {
  std::stringstream ss;
  ss << "SubArenaConfiguration{name_=" << name_ << " blockSize_=" << blockSize_
     << " maxAllocationSize_=" << prettyStringMemorySize(maxAllocationSize_)
     << " relativeSize_=" << relativeSize_ << " allocatedRatioJitThreshold_"
     << allocatedRatioJitThreshold_ << "}";
  return ss.str();
}

std::string MemoryAllocatorConfiguration::prettyString() const {
  std::stringstream ss;
  ss << "MemoryAllocatorConfiguration{name_=" << name_
     << " subArenaConfiguration_.size()=" << subArenaConfiguration_.size()
     << " subArenaConfiguration_={";
  for (const SubArenaConfiguration& subConfig : subArenaConfiguration_) {
    ss << subConfig.prettyString() << ", ";
  }
  ss << "}";
  return ss.str();
}

bool MemoryAllocatorConfiguration::operator<(
    const MemoryAllocatorConfiguration& other) const {
  return alignmentNumberOfBits_ < other.alignmentNumberOfBits_ ||
      subArenaConfiguration_ < other.subArenaConfiguration_;
}

bool MemoryAllocatorConfiguration::operator==(
    const MemoryAllocatorConfiguration& other) const {
  return alignmentNumberOfBits_ == other.alignmentNumberOfBits_ &&
      subArenaConfiguration_ == other.subArenaConfiguration_;
}

bool MemoryAllocatorConfiguration::operator!=(
    const MemoryAllocatorConfiguration& other) const {
  return !(*this == other);
}

void MemoryAllocatorConfiguration::normalize() {
  // Sort by maxAllocationSize_ a->z
  std::sort(
      subArenaConfiguration_.begin(),
      subArenaConfiguration_.end(),
      [](const SubArenaConfiguration& lhs, const SubArenaConfiguration& rhs) {
        return lhs.maxAllocationSize_ < rhs.maxAllocationSize_;
      });
  // We want the end of the composite allocator to catch all allocations. For
  // that, the last allocator that who's size>0 max allocation size should be
  // SIZE_MAX. We iterate from the end of the allocator list and set all last
  // allocators to be catch all, until we set an allocator with size>0 to be
  // a catch all.
  for (int i = subArenaConfiguration_.size() - 1; i >= 0; --i) {
    subArenaConfiguration_[i].maxAllocationSize_ = SIZE_MAX;
    if (subArenaConfiguration_[i].relativeSize_ > 0) {
      break;
    }
  }

  // Ensure block size is a multiple of aligment size and that max allocation
  // size is a multiple of block size.
  const size_t minBlockSize = (1UL << alignmentNumberOfBits_);
  for (const SubArenaConfiguration& subConfig : subArenaConfiguration_) {
    if (subConfig.blockSize_ < minBlockSize ||
        (subConfig.blockSize_ % minBlockSize)) {
      std::stringstream ss;
      ss << "MemoryAllocatorConfiguration::normalize() invalid block size="
         << subConfig.blockSize_
         << " for proper alignment block size must be in multiples of="
         << minBlockSize << " config=" << prettyString();
      FL_LOG(fl::ERROR) << ss.str();
      throw std::runtime_error(ss.str());
    }

    if (subConfig.maxAllocationSize_ < SIZE_MAX &&
        subConfig.maxAllocationSize_ < (subConfig.blockSize_ * 1000)) {
      if ((subConfig.maxAllocationSize_ < subConfig.blockSize_) ||
          (subConfig.maxAllocationSize_ % subConfig.blockSize_)) {
        std::stringstream ss;
        ss << "MemoryAllocatorConfiguration::normalize() invalid maxAllocationSize_="
           << subConfig.maxAllocationSize_
           << " must be in multiples of block size=" << subConfig.blockSize_
           << " config=" << prettyString();
        // FL_LOG(fl::ERROR) << ss.str();
        throw std::runtime_error(ss.str());
      }
    }
  }

  double sum = 0;
  for (const SubArenaConfiguration& subConfig : subArenaConfiguration_) {
    sum += subConfig.relativeSize_;
  }
  for (SubArenaConfiguration& subConfig : subArenaConfiguration_) {
    subConfig.relativeSize_ /= sum;
  }
}

}; // namespace fl
