/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/experimental/memory/allocator/MemoryAllocator.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <utility>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/Utils.h"

namespace fl {

MemoryAllocator::MemoryAllocator(std::string name, int logLevel)
    : name_(std::move(name)), logLevel_(logLevel) {}

const std::string& MemoryAllocator::getName() const {
  return name_;
}

void MemoryAllocator::setName(std::string name) {
  name_ = std::move(name);
}

int MemoryAllocator::getLogLevel() const {
  return logLevel_;
}

void MemoryAllocator::setLogLevel(int logLevel) {
  logLevel_ = logLevel;
}

namespace {

bool compareDouble(double lhs, double rhs) {
  constexpr double epsilon = 1e-3;
  return std::fabs(lhs - rhs) <= epsilon;
}

} // namespace

MemoryAllocator::CommonStats::CommonStats()
    : arenaSize(0), freeCount(0), allocatedCount(0), maxAllocatedCount(0) {}

void MemoryAllocator::CommonStats::allocate(size_t n) {
  allocatedCount += n;
  freeCount -= n;
  maxAllocatedCount = std::max(maxAllocatedCount, allocatedCount);
}

void MemoryAllocator::CommonStats::free(size_t n) {
  allocatedCount -= n;
  freeCount += n;
}

bool MemoryAllocator::CommonStats::operator==(
    const MemoryAllocator::CommonStats& other) const {
  return arenaSize == other.arenaSize && freeCount == other.freeCount &&
      allocatedCount == other.allocatedCount &&
      maxAllocatedCount == other.maxAllocatedCount;
}

bool MemoryAllocator::CommonStats::operator!=(
    const MemoryAllocator::CommonStats& other) const {
  return !(*this == other);
}

std::string MemoryAllocator::CommonStats::diffPrettyString(
    const CommonStats& other) const {
  std::stringstream ss;
  if (arenaSize != other.arenaSize) {
    ss << " arenaSize={this=" << arenaSize << " other=" << other.arenaSize
       << "}";
  }
  if (allocatedCount != other.allocatedCount) {
    ss << " allocatedCount={this=" << allocatedCount
       << " other=" << other.allocatedCount << "}";
  }
  if (maxAllocatedCount != other.maxAllocatedCount) {
    ss << " maxAllocatedCount={this=" << maxAllocatedCount
       << " other=" << other.maxAllocatedCount << "}";
  }
  return ss.str();
}

double MemoryAllocator::CommonStats::allocatedRatio() const {
  if (arenaSize == 0) {
    return 0;
  }
  return static_cast<double>(allocatedCount) / static_cast<double>(arenaSize);
}

double MemoryAllocator::CommonStats::maxAllocatedRatio() const {
  if (arenaSize == 0) {
    return 0;
  }
  return static_cast<double>(maxAllocatedCount) /
      static_cast<double>(arenaSize);
}

std::string MemoryAllocator::CommonStats::prettyString() const {
  std::stringstream stream;
  stream << "arenaSize=" << prettyStringMemorySize(arenaSize)
         << " freeCount=" << prettyStringMemorySize(freeCount)
         << " allocatedCount=" << prettyStringMemorySize(allocatedCount)
         << " allocatedRatio=" << allocatedRatio()
         << " maxAllocatedCount=" << prettyStringMemorySize(maxAllocatedCount)
         << " maxAllocatedRatio=" << maxAllocatedRatio() << "}";

  return stream.str();
}

bool MemoryAllocator::Stats::operator==(
    const MemoryAllocator::Stats& other) const {
  return blockSize == other.blockSize &&
      allocationsCount == other.allocationsCount &&
      deAllocationsCount == other.deAllocationsCount &&
      compareDouble(
             internalFragmentationScore, other.internalFragmentationScore) &&
      compareDouble(
             externalFragmentationScore, other.externalFragmentationScore) &&
      statsInBytes == other.statsInBytes &&
      statsInBlocks == other.statsInBlocks;
}

bool MemoryAllocator::Stats::operator!=(
    const MemoryAllocator::Stats& other) const {
  return !(*this == other);
}

std::string MemoryAllocator::Stats::diffPrettyString(const Stats& other) const {
  std::stringstream ss;
  if (blockSize != other.blockSize) {
    ss << " blockSize={this=" << blockSize << " other=" << other.blockSize
       << "}";
  }
  if (allocationsCount != other.allocationsCount) {
    ss << " allocationsCount={this=" << allocationsCount
       << " other=" << other.allocationsCount << "}";
  }
  if (deAllocationsCount != other.deAllocationsCount) {
    ss << " deAllocationsCount={this=" << deAllocationsCount
       << " other=" << other.deAllocationsCount << "}";
  }
  if (!compareDouble(
          internalFragmentationScore, other.internalFragmentationScore)) {
    ss << " internalFragmentationScore={this=" << internalFragmentationScore
       << " other=" << other.internalFragmentationScore << "}";
  }
  if (!compareDouble(
          externalFragmentationScore, other.externalFragmentationScore)) {
    ss << " externalFragmentationScore={this=" << externalFragmentationScore
       << " other=" << other.externalFragmentationScore << "}";
  }
  if (statsInBytes != other.statsInBytes) {
    ss << " statsInBytes={" << statsInBytes.diffPrettyString(other.statsInBytes)
       << "}";
  }
  if (statsInBlocks != other.statsInBlocks) {
    ss << " statsInBlocks={"
       << statsInBlocks.diffPrettyString(other.statsInBlocks) << "}";
  }
  return ss.str();
}

std::string MemoryAllocator::Stats::prettyString() const {
  std::stringstream stream;
  stream << "Stats{" << std::endl;
  stream << "highlights={size="
         << prettyStringMemorySize(statsInBytes.arenaSize) << ' ';
  if (oomEventCount > 0) {
    stream << "oomEventCount=" << oomEventCount << ' ';
  }
  if (statsInBlocks.maxAllocatedRatio() > 0.5) {
    stream << "maxAllocatedRatio=" << statsInBlocks.maxAllocatedRatio() << ' ';
  }
  if (maxInternalFragmentationScore > 0.25) {
    stream << "maxInternalFragmentationScore=" << maxInternalFragmentationScore
           << ' ';
  }
  if (maxExternalFragmentationScore > 0.25) {
    stream << "maxExternalFragmentationScore=" << maxExternalFragmentationScore
           << ' ';
  }
  stream << "currentlyAllocatedCnt="
         << prettyStringCount(allocationsCount - deAllocationsCount) << '}';

  stream << " arena=" << arena
         << " blockSize=" << prettyStringMemorySize(blockSize)
         << " allocationsCount=" << prettyStringCount(allocationsCount)
         << " deAllocationsCount=" << prettyStringCount(deAllocationsCount)
         << " internalFragmentationScore=" << internalFragmentationScore
         << " externalFragmentationScore=" << externalFragmentationScore
         << " maxInternalFragmentationScore=" << maxInternalFragmentationScore
         << " maxExternalFragmentationScore=" << maxExternalFragmentationScore
         << " oomEventCount=" << oomEventCount
         << " performanceCost=" << prettyStringCount(performanceCost)
         << std::endl
         << " statsInBytes={" << statsInBytes.prettyString() << std::endl
         << "} statsInBlocks={" << statsInBlocks.prettyString() << '}'
         << std::endl;
  if (!subArenaStats.empty()) {
    stream << "\nsubArenaStats={";
    for (const Stats& subStats : subArenaStats) {
      stream << '\n' << subStats.prettyString();
    }
    stream << "\n}";
  }
  stream << '}';
  return stream.str();
} // namespace fl

MemoryAllocator::Stats::Stats() : Stats(nullptr, 0, 0) {}

MemoryAllocator::Stats::Stats(
    void* arena,
    size_t arenaSizeInBytes,
    size_t blockSize)
    : arena(arena),
      blockSize(blockSize),
      allocationsCount(0),
      deAllocationsCount(0),
      internalFragmentationScore(0.0),
      externalFragmentationScore(0.0),
      maxInternalFragmentationScore(0.0),
      maxExternalFragmentationScore(0.0),
      oomEventCount(0),
      performanceCost(0),
      failToAllocate(false) {
  if (arenaSizeInBytes > 0) {
    statsInBytes.arenaSize = arenaSizeInBytes;
    statsInBytes.freeCount = arenaSizeInBytes;
    if (blockSize > 0) {
      statsInBlocks.arenaSize = arenaSizeInBytes / blockSize;
      statsInBlocks.freeCount = arenaSizeInBytes / blockSize;
    }
  }
}

void MemoryAllocator::Stats::incrementAllocationsCount() {
  ++allocationsCount;
}

void MemoryAllocator::Stats::incrementDeAllocationsCount() {
  ++deAllocationsCount;
}

void MemoryAllocator::Stats::setExternalFragmentationScore(double score) {
  externalFragmentationScore = score;
  maxExternalFragmentationScore =
      std::max(maxExternalFragmentationScore, score);
}

void MemoryAllocator::Stats::incrementOomEventCount() {
  ++oomEventCount;
}

void MemoryAllocator::Stats::addPerformanceCost(size_t cost) {
  if (performanceCost == ULLONG_MAX) {
    return;
  }
  if (ULLONG_MAX - cost > performanceCost) {
    performanceCost += cost;
  } else {
    FL_LOG(fl::WARNING)
        << "MemoryAllocator::Stats::addPerformanceCost(cost=" << cost
        << ") reach maximum cost and will not be tracked any longer.";
    performanceCost = ULLONG_MAX;
  }
}

namespace {
double calcInternalFragmentationScore(
    double allocatedBytes,
    double allocatedBlocks,
    double blockSize) {
  const double bytesAllocatedOutOfBlocksAllocated = (allocatedBytes != 0)
      ? (allocatedBytes / (allocatedBlocks * blockSize))
      : 1.0;

  return 1.0 - bytesAllocatedOutOfBlocksAllocated;
}
} // namespace

void MemoryAllocator::Stats::allocate(size_t bytes, size_t blocks) {
  ++allocationsCount;
  statsInBytes.allocate(bytes);
  statsInBlocks.allocate(blocks);
  internalFragmentationScore = calcInternalFragmentationScore(
      static_cast<double>(statsInBytes.allocatedCount),
      static_cast<double>(statsInBlocks.allocatedCount),
      static_cast<double>(blockSize));
  maxInternalFragmentationScore =
      std::max(maxInternalFragmentationScore, internalFragmentationScore);
}

void MemoryAllocator::Stats::free(size_t bytes, size_t blocks) {
  ++deAllocationsCount;
  statsInBytes.free(bytes);
  statsInBlocks.free(blocks);
  internalFragmentationScore = calcInternalFragmentationScore(
      static_cast<double>(statsInBytes.allocatedCount),
      static_cast<double>(statsInBlocks.allocatedCount),
      static_cast<double>(blockSize));
}
}; // namespace fl
