/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/memory/allocator/ConfigurableMemoryAllocator.h"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <utility>

#include "flashlight/common/CppBackports.h"
#include "flashlight/common/Utils.h"
#include "flashlight/memory/allocator/CompositeMemoryAllocator.h"
#include "flashlight/memory/allocator/freelist/FreeList.h"
#include "flashlight/memory/allocator/memorypool/MemoryPool.h"

namespace fl {

std::unique_ptr<MemoryAllocator> CreateMemoryAllocator(
    const MemoryAllocatorConfiguration& config) {
  double sumRelativeSize = 0.0;
  for (const SubArenaConfiguration& subArenaConfig :
       config.subArenaConfiguration) {
    sumRelativeSize += subArenaConfig.relativeSize;
  }

  auto compositeAllocator = fl::cpp::make_unique<CompositeMemoryAllocator>();
  void* subArenaAddress = config.arena;
  for (const SubArenaConfiguration& subArenaConfig :
       config.subArenaConfiguration) {
    // Ensure block size matches alignmnet.
    if (!(subArenaConfig.blockSize >> config.alignmentNumberOfBits)) {
      std::stringstream ss;
      ss << "CreateMemoryAllocator() invalid block size="
         << subArenaConfig.blockSize
         << " for proper alignment block size must be in multiples of="
         << (1 << config.alignmentNumberOfBits);
      throw std::invalid_argument(ss.str());
    }

    const size_t maxSubArenaSize = config.arenaSizeInBytes -
        (static_cast<char*>(subArenaAddress) -
         static_cast<char*>(config.arena));
    size_t subAreanSize = static_cast<size_t>(
        static_cast<double>(config.arenaSizeInBytes) *
        (subArenaConfig.relativeSize * sumRelativeSize));

    // Round up allocator arena size to next block.
    const size_t nBlocks = divRoundUp(subAreanSize, subArenaConfig.blockSize);
    subAreanSize = nBlocks * subArenaConfig.blockSize;

    subAreanSize = std::min(subAreanSize, maxSubArenaSize);

    std::unique_ptr<MemoryAllocator> subAllocator;
    if (subArenaConfig.blockSize < subArenaConfig.maxAllocationSize) {
      subAllocator = fl::cpp::make_unique<FreeList>(
          subArenaConfig.name,
          subArenaAddress,
          subAreanSize,
          subArenaConfig.blockSize);
    } else {
      subAllocator = fl::cpp::make_unique<MemoryPool>(
          subArenaConfig.name,
          subArenaAddress,
          subAreanSize,
          subArenaConfig.blockSize);
    }
    compositeAllocator->add(
        {subArenaConfig.maxAllocationSize, std::move(subAllocator)});
    subArenaAddress = static_cast<char*>(subArenaAddress) + subAreanSize;
  }
  return std::move(compositeAllocator);
}

}; // namespace fl
