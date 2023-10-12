/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/dataset/BlobDataset.h"

#include <fstream>
#include <mutex>

namespace fl {

/**
 * A BlobDataset on file.
 *
 * As the arrays are stored on disk, sequential access will be the most
 * efficient.
 *
 */
class FL_API FileBlobDataset : public BlobDataset {
 public:
  /**
   * Creates a `FileBlobDataset`, specifying a blob file name.
   * @param[in] name A blob file name.
   * @param[in] rw If true, opens in read-write mode. This must be specified
   * to use the add() and synch() methods. Except if truncate is true,
   * previous stored samples will be read.
   * @param[in] truncate In read-write mode, truncate the files if it
   * already exists.
   */
  explicit FileBlobDataset(
      const fs::path& name,
      bool rw = false,
      bool truncate = false);

  virtual ~FileBlobDataset() override;

 protected:
  int64_t writeData(int64_t offset, const char* data, int64_t size)
      const override;
  int64_t readData(int64_t offset, char* data, int64_t size) const override;
  void flushData() override;
  bool isEmptyData() const override;

 private:
  fs::path name_;
  std::ios_base::openmode mode_;
  std::shared_ptr<std::fstream> getStream() const;

  mutable std::vector<std::weak_ptr<
      std::unordered_map<uintptr_t, std::shared_ptr<std::fstream>>>>
      allFileHandles_;
  mutable std::mutex afhmutex_;
};

} // namespace fl
