/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/BlobDataset.h"

#include <fstream>
#include <mutex>

namespace fl {

/**
 * A BlobDataset in (CPU) memory.
 *
 * As the arrays are stored on disk, sequential access will be the most
 * efficient.
 *
 */
class MemoryBlobDataset : public BlobDataset {
 public:
  /**
   * Creates a `MemoryBlobDataset`, specifying a blob file name.
   */
  MemoryBlobDataset();

  virtual ~MemoryBlobDataset() override = default;

 protected:
  int64_t writeData(int64_t offset, const char* data, int64_t size)
      const override;
  int64_t readData(int64_t offset, char* data, int64_t size) const override;
  void flushData() override;
  bool isEmptyData() const override;

 private:
  mutable std::mutex writeMutex_;
  mutable std::vector<char> data_;
};

} // namespace fl
