/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <stdexcept>
#include <thread>

#include "flashlight/fl/dataset/BlobDataset.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

const int64_t magicNumber = 0x31626f6c423a6c66;

BlobDatasetEntryBuffer::BlobDatasetEntryBuffer() = default;

void BlobDatasetEntryBuffer::clear() {
  data_.clear();
}

int64_t BlobDatasetEntryBuffer::size() const {
  return data_.size() / nFieldPerEntry_;
}

void BlobDatasetEntryBuffer::resize(int64_t size) {
  data_.resize(size * nFieldPerEntry_);
}

BlobDatasetEntry BlobDatasetEntryBuffer::get(const int64_t idx) const {
  BlobDatasetEntry e;
  auto dataIdx = idx * nFieldPerEntry_;
  e.type = static_cast<fl::dtype>(data_[dataIdx++]);
  unsigned numDims = data_[dataIdx++];
  e.dims = Shape(std::vector<Dim>(numDims));
  for (int i = 0; i < numDims; i++) {
    e.dims[i] = data_[dataIdx + i];
  }
  e.offset = data_[dataIdx + maxNDims_];
  return e;
}

void BlobDatasetEntryBuffer::add(const BlobDatasetEntry& e) {
  data_.push_back(static_cast<int64_t>(e.type));
  data_.push_back(static_cast<int64_t>(e.dims.ndim()));
  int i = 0;
  for (; i < e.dims.ndim(); i++) {
    data_.push_back(e.dims[i]);
  }
  for (; i < maxNDims_; ++i) {
    data_.push_back(1); // placeholder dim
  }
  data_.push_back(e.offset);
}

char* BlobDatasetEntryBuffer::data() {
  return (char*)data_.data();
}

int64_t BlobDatasetEntryBuffer::bytes() const {
  return data_.size() * sizeof(int64_t);
};

BlobDataset::BlobDataset() = default;

int64_t BlobDataset::size() const {
  return offsets_.size();
}

std::vector<Tensor> BlobDataset::get(const int64_t idx) const {
  std::vector<Tensor> sample;
  for (int64_t i = 0; i < sizes_.at(idx); i++) {
    auto entry = entries_.get(offsets_.at(idx) + i);
    sample.push_back(readArray(entry, i));
  }
  return sample;
};

std::vector<std::vector<uint8_t>> BlobDataset::rawGet(const int64_t idx) const {
  std::vector<std::vector<uint8_t>> sample;
  for (int64_t i = 0; i < sizes_.at(idx); i++) {
    auto entry = entries_.get(offsets_.at(idx) + i);
    sample.push_back(readRawArray(entry));
  }
  return sample;
};

void BlobDataset::add(const std::vector<Tensor>& sample) {
  int64_t entryOffset;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    entryOffset = entries_.size();
    offsets_.push_back(entries_.size());
    sizes_.push_back(sample.size());
    for (const auto& tensor : sample) {
      if (tensor.ndim() > maxNDims_) {
        throw std::invalid_argument(
            "BlobDataset::add - no support for serialization of "
            "tensors with > 4 dimensions");
      }
      BlobDatasetEntry e;
      e.type = tensor.type();
      e.dims = tensor.shape();
      e.offset = indexOffset_;
      indexOffset_ += tensor.bytes();
      entries_.add(e);
    }
  }
  for (int64_t i = 0; i < sample.size(); i++) {
    auto& array = sample[i];
    const auto& e = entries_.get(entryOffset + i);
    writeArray(e, array);
  }
}

void BlobDataset::add(const BlobDataset& blob, int64_t chunkSize) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (chunkSize <= 0) {
    throw std::runtime_error("chunkSize must be positive");
  }
  sizes_.insert(sizes_.end(), blob.sizes_.begin(), blob.sizes_.end());
  std::vector<int64_t> offsets = blob.offsets_;
  for (auto& offset : offsets) {
    offset += entries_.size();
  }
  offsets_.insert(offsets_.end(), offsets.begin(), offsets.end());
  for (int64_t i = 0; i < blob.entries_.size(); i++) {
    auto e = blob.entries_.get(i);
    e.offset += indexOffset_ - 2 * sizeof(int64_t);
    entries_.add(e);
  }
  int64_t blobOffset = 2 * sizeof(int64_t);
  int64_t copySize = blob.indexOffset_ - blobOffset;
  int64_t nChunk = copySize / chunkSize;
  int64_t remainCopySize = copySize - nChunk * chunkSize;
  std::vector<char> buffer;
  auto copyChunk = [&buffer, &blob, this, &blobOffset](int64_t size) {
    buffer.resize(size);
    blob.readData(blobOffset, buffer.data(), size);
    blobOffset += size;
    this->writeData(indexOffset_, buffer.data(), size);
    this->indexOffset_ += size;
  };
  for (int64_t i = 0; i < nChunk; i++) {
    copyChunk(chunkSize);
  }
  if (remainCopySize > 0) {
    copyChunk(remainCopySize);
  }
}

std::vector<uint8_t> BlobDataset::readRawArray(
    const BlobDatasetEntry& e) const {
  std::vector<uint8_t> buffer;
  if (e.dims.elements() > 0) {
    buffer.resize(fl::getTypeSize(e.type) * e.dims.elements());
    readData(
        e.offset,
        (char*)buffer.data(),
        fl::getTypeSize(e.type) * e.dims.elements());
  }
  return buffer;
}

Tensor BlobDataset::readArray(const BlobDatasetEntry& e, int i) const {
  if (e.dims.elements() > 0) {
    auto buffer = readRawArray(e);
    auto keyval = hostTransforms_.find(i);
    if (keyval == hostTransforms_.end()) {
      return Tensor::fromBuffer(
          e.dims, e.type, buffer.data(), MemoryLocation::Host);
    } else {
      return keyval->second(buffer.data(), e.dims, e.type);
    }
  } else {
    return Tensor();
  }
}

void BlobDataset::writeArray(const BlobDatasetEntry& e, const Tensor& array) {
  std::vector<uint8_t> buffer(array.bytes());
  array.host(buffer.data());
  writeData(e.offset, (char*)buffer.data(), buffer.size());
}

void BlobDataset::writeIndex() {
  std::lock_guard<std::mutex> lock(mutex_);

  int64_t offset = 0;
  offset += writeData(offset, (char*)&magicNumber, sizeof(int64_t));
  writeData(offset, (char*)&indexOffset_, sizeof(int64_t));

  offset = indexOffset_;
  int64_t size = offsets_.size();
  int64_t entriesSize = entries_.size();
  offset += writeData(offset, (char*)&size, sizeof(int64_t));
  offset += writeData(offset, (char*)&entriesSize, sizeof(int64_t));
  offset += writeData(offset, (char*)sizes_.data(), sizeof(int64_t) * size);
  offset += writeData(offset, (char*)offsets_.data(), sizeof(int64_t) * size);
  writeData(offset, entries_.data(), entries_.bytes());
  flushData();
}

void BlobDataset::readIndex() {
  std::lock_guard<std::mutex> lock(mutex_);

  entries_.clear();

  if (isEmptyData()) {
    // skip magic number and index location
    indexOffset_ = 2 * sizeof(int64_t);
    return;
  }

  int64_t magicNumberCheck = 0;
  int64_t offset = readData(0, (char*)&magicNumberCheck, sizeof(int64_t));
  if (magicNumber != magicNumberCheck) {
    throw std::runtime_error("BlobDataset::readIndex - not a fl::BlobDataset");
  }
  readData(offset, (char*)&indexOffset_, sizeof(int64_t));
  offset = indexOffset_;

  int64_t size;
  int64_t entriesSize;
  offset += readData(offset, (char*)&size, sizeof(int64_t));
  offset += readData(offset, (char*)&entriesSize, sizeof(int64_t));
  sizes_.resize(size);
  offsets_.resize(size);
  entries_.resize(entriesSize);

  offset += readData(offset, (char*)sizes_.data(), sizeof(int64_t) * size);
  offset += readData(offset, (char*)offsets_.data(), sizeof(int64_t) * size);
  readData(offset, entries_.data(), entries_.bytes());
}

void BlobDataset::flush() {
  flushData();
}

void BlobDataset::setHostTransform(
    int field,
    std::function<Tensor(void*, fl::Shape, fl::dtype)> func) {
  hostTransforms_[field] = func;
}

std::vector<BlobDatasetEntry> BlobDataset::getEntries(const int64_t idx) const {
  std::vector<BlobDatasetEntry> entries;
  for (int64_t i = 0; i < sizes_.at(idx); i++) {
    entries.push_back(entries_.get(offsets_.at(idx) + i));
  }
  return entries;
}

BlobDataset::~BlobDataset() = default;

} // namespace fl
