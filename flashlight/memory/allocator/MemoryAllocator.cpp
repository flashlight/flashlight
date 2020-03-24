/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iostream>
#include <sstream>
#include <utility>

#include "flashlight/memory/allocator/MemoryAllocator.h"

namespace fl {

MemoryAllocator::MemoryAllocator(std::string name) : name_(std::move(name)) {}

const std::string& MemoryAllocator::getName() const {
  return name_;
}

namespace {

void formatMemorysize(std::stringstream& stream, size_t size) {
  stream << size;
  if (size >= (1L << 43)) { // >= 8TB
    stream << '(' << (size >> 40) << "TB)";
  } else if (size >= (1L << 33)) { // >= 8G B
    stream << '(' << (size >> 30) << "GB)";
  } else if (size >= (1L << 23)) { // >= 8M B
    stream << '(' << (size >> 20) << "MB)";
  } else if (size >= (1L << 13)) { // >= 8K B
    stream << '(' << (size >> 10) << "KB)";
  }
}

void formatCount(std::stringstream& stream, size_t count) {
  stream << count;
  if (count >= 10e13) { // >= 10 trillion
    stream << '(' << (count / (size_t)10e12) << "t)";
  } else if (count >= 10e10) { // >= 10 billion
    stream << '(' << (count / (size_t)10e9) << "b)";
  } else if (count >= 10e7) { // >= 10 million
    stream << '(' << (count / (size_t)10e6) << "m)";
  } else if (count >= 10e4) { // >= 10 thousand
    stream << '(' << (count / (size_t)10e3) << "k)";
  }
}

} // namespace

std::string MemoryAllocator::CommonStats::prettyString() const {
  std::stringstream stream;

  stream << "arenaSize=";
  formatMemorysize(stream, arenaSize);
  stream << " freeCount=";
  formatMemorysize(stream, freeCount);
  stream << " allocatedCount=";
  formatMemorysize(stream, allocatedCount);
  stream << " allocatedRatio=" << allocatedRatio << "}";

  return stream.str();
}

std::string MemoryAllocator::Stats::prettyString() const {
  std::stringstream stream;

  stream << "Stats{arena=" << arena << " blockSize=";
  formatMemorysize(stream, blockSize);
  stream << " allocationsCount=";
  formatCount(stream, allocationsCount);
  stream << " deAllocationsCount=";
  formatCount(stream, deAllocationsCount);
  stream << " internalFragmentationScore=" << internalFragmentationScore
         << " externalFragmentationScore=" << externalFragmentationScore
         << " statsInBytes={" << statsInBytes.prettyString();
  stream << "} statsInBlocks={" << statsInBlocks.prettyString();
  stream << "}}";

  return stream.str();
}

}; // namespace fl
