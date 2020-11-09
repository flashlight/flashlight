/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include "flashlight/fl/dataset/MemoryBlobDataset.h"

namespace fl {

MemoryBlobDataset::MemoryBlobDataset() {
  readIndex();
}

int64_t MemoryBlobDataset::writeData(
    int64_t offset,
    const char* data,
    int64_t size) const {
  std::lock_guard<std::mutex> lock(writeMutex_);
  if (offset + size > data_.size()) {
    data_.resize(offset + size);
  }
  std::memcpy(data_.data() + offset, data, size);
  return size;
}

int64_t MemoryBlobDataset::readData(int64_t offset, char* data, int64_t size)
    const {
  // what is available
  int64_t maxSize = std::max(0UL, data_.size() - offset);
  // min(what is available, wanted)
  maxSize = std::min(maxSize, size);
  std::memcpy(data, data_.data() + offset, maxSize);
  return maxSize;
}

void MemoryBlobDataset::flushData() {
  std::lock_guard<std::mutex> lock(writeMutex_);
}

bool MemoryBlobDataset::isEmptyData() const {
  return (data_.size() == 0);
}

} // namespace fl
