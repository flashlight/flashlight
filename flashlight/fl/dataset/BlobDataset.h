/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/Dataset.h"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace fl {

/**
 * An abstract dataset storing af::arrays in a binary archive (blob). A
 * corresponding index is built, which stores all array positions, which
 * allows efficient array access from the blob. Both array data and index
 * are stored in a single blob.
 *
 * The dataset allows heterogenous storage: the number of arrays per sample
 * is not constrained (different samples can have different number of
 * arrays). Stored arrays can be also of any type.
 *
 * The dataset should be first created by opening a writable blob
 * (implementation dependent), then using the add() function. Datasets
 * opened in write mode can be read too, assuming the data written in the
 * binary archive has been flushed with flush(). Before any subsequent
 * openings, the blob index must be written with writeIndex().
 *
 * The dataset is thread-safe for read and write operations.
 *
 * Concrete versions of this class must implement writeData(), readData(),
 * flushData() and isEmptyData().
 *
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
  <int64*6*entries: entry table seen as int64s {type, dim0, .. dim3, offset}>
  \endcode
  *
 */

struct BlobDatasetEntry {
  af::dtype type;
  af::dim4 dims;
  int64_t offset;
};

class BlobDatasetEntryBuffer {
 private:
  std::vector<int64_t> data_;
  const int nFieldPerEntry_ = 6;

 public:
  BlobDatasetEntryBuffer();
  void clear();
  int64_t size() const;
  void resize(int64_t size);
  BlobDatasetEntry get(const int64_t idx) const;
  void add(const BlobDatasetEntry& entry);
  char* data();
  int64_t bytes() const;
};

class BlobDataset : public Dataset {
 private:
  BlobDatasetEntryBuffer entries_;
  std::vector<int64_t> sizes_;
  std::vector<int64_t> offsets_;
  int64_t indexOffset_;
  std::unordered_map<int, DataTransformFunction> hostTransforms_;
  mutable std::mutex mutex_;

  std::vector<uint8_t> readRawArray(const BlobDatasetEntry& e) const;
  af::array readArray(const BlobDatasetEntry& e, int i) const;
  void writeArray(const BlobDatasetEntry& e, const af::array& array);

 protected:
  void readIndex();

  /* Write raw data in the blob.
   * Implementation must be thread-safe.
   * @param[in] offset Offset in the blob in bytes.
   * @param[in] data Raw data bytes.
   * @param[in] size Raw data size in bytes.
   */
  virtual int64_t writeData(int64_t offset, const char* data, int64_t size)
      const = 0;
  /* Read raw data in the blob.
   * Implementation must be thread-safe.
   * @param[in] offset Offset in the blob in bytes.
   * @param[out] data Raw data bytes.
   * @param[in] size Raw data size in bytes.
   */
  virtual int64_t readData(int64_t offset, char* data, int64_t size) const = 0;
  /* Make sure all written data is flushed in the blob.
   * Implementation must be thread-safe.
   */
  virtual void flushData() = 0;
  /* Return true iff the blob is empty.
   * Implementation must be thread-safe.
   */
  virtual bool isEmptyData() const = 0;

 public:
  /**
   * Creates a `BlobDataset`, specifying a blob file name.
   * @param[in] name A blob file name.
   * @param[in] rw If true, opens in read-write mode. This must be specified
   * to use the add() and writeIndex() methods. Except if truncate is true,
   * previous stored samples will be read.
   * @param[in] truncate In read-write mode, truncate the files if it
   * already exists.
   */
  BlobDataset();

  int64_t size() const override;

  std::vector<af::array> get(const int64_t idx) const override;

  /**
   * Return raw data stored in given sample. Dimensions and types of each array
   * can be retrieved with getEntries().
   * @param[in] idx An index in the dataset.
   */
  std::vector<std::vector<uint8_t>> rawGet(const int64_t idx) const;

  /**
   * Add a new sample in the dataset. The dataset must have been opened in
   * read-write mode. Data is guaranteed to be on disk only after a flush().
   * @param[in] sample A vector of arrays, possibly of heterogeneous types and
   * sizes.
   */
  void add(const std::vector<af::array>& sample);

  /**
   * Add an entire blob to the current blob. This efficiently concatenate
   * blobs by reading and writing (possibly large) chunks.
   * @param[in] blob The blob to be added.
   * @param[in] chunkSize Read-write chunk size.
   */
  void add(const BlobDataset& blob, int64_t chunkSize = 104857600);

  /**
   * Flush all data on disk. The dataset must have been opened in
   * read-write mode.
   */
  void flush();

  /**
   * Write index and flush data.
   */
  void writeIndex();

  /**
   * Set a host transform on specified field. If a host transform is
   * specified, it will be called to load the data from host to af::array
   * (on device).
   * @param[in] field The field on which to apply the transform.
   * @param[in] func The corresponding transform.
   */
  void setHostTransform(
      int field,
      std::function<af::array(void*, af::dim4, af::dtype)> func);

  /** Return entries in the blob for a given sample index.
   *  @param[in] idx A sample index.
   */
  std::vector<BlobDatasetEntry> getEntries(const int64_t idx) const;

  virtual ~BlobDataset() override;
};

} // namespace fl
