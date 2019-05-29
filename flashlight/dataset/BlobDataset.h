/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/dataset/Dataset.h"

#include <fstream>
#include <mutex>

namespace fl {

/**
 * A dataset storing af::arrays on disk, concatenating all arrays in a
 * binary archive. A corresponding index is built, which stores all array
 * positions, which allows efficient array access from the binary archive.
 * Both the binary archive and its index are stored in a single file.
 *
 * As the arrays are stored on disk, sequential access will be the most
 * efficient.
 *
 * The dataset allows heterogenous storage: the number of arrays per sample
 * is not constrained (different samples can have different number of
 * arrays). Stored arrays can be also of any type.
 *
 * The dataset should be first created by opening the blob file in write
 * mode, then using the add() function. Datasets opened in write mode can
 * be read too, assuming the data written in the binary archive has been
 * flushed. Before any subsequent openings, the blob must be synchronized
 * with sync().
 *
 * The dataset is thread-safe for read and write operations.
 *
 * Example:
  \code{.cpp}
  af::array tensor1 = af::randu(5, 4, 10);
  af::array tensor2 = af::randu(7, 10);
  af::array tensor3 = af::randu(2, 4);

  // Create the dataset
  BlobDataset dsrw("archive.blob", true, true);
  ds.add({tensor1});
  ds.add({tensor2, tensor3});
  ds.sync();

  // Read it
  BlobDataset ds("archive.blob");
  for(auto& sample : ds) {
    std::cout << "sample size: " << sample.size() << std::cout;
  }
  \endcode
  *
  * For advanced users, the format of the blob is the following:
  \code{.unparsed}
  <int64: magic number (0x31626f6c423a6c66)>
  <int64: offset to index>
  ---- raw data ----
  <raw tensor data>
  ...
  <raw tensor data>
  ---- index ----
  <int64: # of samples in dataset (size)>
  <int64: # of tensors in dataset (entries)>
  <int64*size: number of arrays per sample>
  <int64*size: start offset in entry table for each sample>
  <int64*6*entries: entry table: description of each array entry>
  \endcode
  *
 */

struct BlobDatasetEntry {
  af::dtype type;
  af::dim4 dims;
  int64_t offset;
  void write(std::ostream& file) const;
  void read(std::istream& file);
};

class BlobDataset : public Dataset {
 private:
  std::vector<BlobDatasetEntry> entries_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> offsets_;
  mutable std::fstream fs_;
  mutable std::mutex mutex_;
  int64_t indexOffset_;

  af::array readArray(const BlobDatasetEntry& e) const;
  BlobDatasetEntry writeArray(const af::array& array);
  void readIndex();

 public:
  /**
   * Creates a `BlobDataset`, specifying a blob file name.
   * @param[in] name A blob file name.
   * @param[in] rw If true, opens in read-write mode. This must be specified
   * to use the add() and synch() methods. Except if truncate is true,
   * previous stored samples will be read.
   * @param[in] truncate In read-write mode, truncate the files if it
   * already exists.
   */
  explicit BlobDataset(
      const std::string& name,
      bool rw = false,
      bool truncate = false);

  int64_t size() const override;

  std::vector<af::array> get(const int64_t idx) const override;

  /**
   * Add a new sample in the dataset. The dataset must have been opened in
   * read-write mode. Data is guaranteed to be on disk only after a sync().
   * @param[in] sample A vector of arrays, possibly of heterogeneous types and
   * sizes.
   */
  void add(const std::vector<af::array>& sample);

  /**
   * Synchronize all data on disk. The dataset must have been opened in
   * read-write mode.
   */
  void sync();
};

} // namespace fl
