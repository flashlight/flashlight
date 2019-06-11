/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <array>
#include <thread>

#include "flashlight/dataset/BlobDataset.h"

namespace fl {

const int64_t magicNumber = 0x31626f6c423a6c66;

void BlobDatasetEntry::write(std::ostream& file) const {
  std::array<int64_t, 6> data;
  data[0] = static_cast<int64_t>(type);
  for (int i = 0; i < 4; i++) {
    data[i + 1] = dims[i];
  }
  data[5] = offset;
  file.write((char*)&data, sizeof(int64_t) * 6);
}

void BlobDatasetEntry::read(std::istream& file) {
  std::array<int64_t, 6> data;
  file.read((char*)&data, sizeof(int64_t) * 6);
  type = static_cast<af::dtype>(data[0]);
  for (int i = 0; i < 4; i++) {
    dims[i] = data[i + 1];
  }
  offset = data[5];
}

BlobDataset::BlobDataset(const std::string& name, bool rw, bool truncate) {
  std::ios_base::openmode mode =
      (rw ? std::ios_base::in | std::ios_base::out : std::ios_base::in);
  if (rw && truncate) {
    mode |= std::ios_base::trunc;
  }
  fs_.exceptions(
      std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
  fs_.open(name, mode);
  readIndex();
}

int64_t BlobDataset::size() const {
  return offsets_.size();
}

std::vector<af::array> BlobDataset::get(const int64_t idx) const {
  std::vector<af::array> sample;
  for (int64_t i = 0; i < sizes_.at(idx); i++) {
    auto entry = entries_.at(offsets_.at(idx) + i);
    sample.push_back(readArray(entry));
  }
  return sample;
};

void BlobDataset::add(const std::vector<af::array>& sample) {
  std::lock_guard<std::mutex> lock(mutex_);
  offsets_.push_back(entries_.size());
  sizes_.push_back(sample.size());
  for (const auto& array : sample) {
    auto e = writeArray(array);
    entries_.push_back(e);
  }
}

af::array BlobDataset::readArray(const BlobDatasetEntry& e) const {
  if (e.dims.elements() > 0) {
    std::vector<uint8_t> buffer(af::getSizeOf(e.type) * e.dims.elements());
    {
      std::lock_guard<std::mutex> lock(mutex_);
      fs_.seekg(e.offset, std::ios_base::beg);
      fs_.read((char*)buffer.data(), af::getSizeOf(e.type) * e.dims.elements());
    }
    af_array c_array;
    af_err status = af_create_array(
        &c_array, buffer.data(), e.dims.ndims(), e.dims.get(), e.type);
    if (status != AF_SUCCESS) {
      throw af::exception("unable to create array", __FILE__, __LINE__, status);
    }
    return af::array(c_array);
  } else {
    return af::array();
  }
}

BlobDatasetEntry BlobDataset::writeArray(const af::array& array) {
  BlobDatasetEntry e;
  e.type = array.type();
  e.dims = array.dims();
  e.offset = indexOffset_;
  std::vector<uint8_t> buffer(array.bytes());
  array.host(buffer.data());
  fs_.seekg(e.offset, std::ios_base::beg);
  fs_.write((char*)buffer.data(), af::getSizeOf(e.type) * e.dims.elements());
  indexOffset_ = fs_.tellp();
  return e;
}

void BlobDataset::sync() {
  std::lock_guard<std::mutex> lock(mutex_);

  fs_.seekp(0, std::ios_base::beg);
  fs_.write((char*)&magicNumber, sizeof(int64_t));
  fs_.write((char*)&indexOffset_, sizeof(int64_t));
  fs_.seekp(indexOffset_, std::ios_base::beg);

  int64_t size = offsets_.size();
  int64_t entries_size = entries_.size();
  fs_.write((char*)&size, sizeof(int64_t));
  fs_.write((char*)&entries_size, sizeof(int64_t));
  fs_.write((char*)sizes_.data(), sizeof(int64_t) * size);
  fs_.write((char*)offsets_.data(), sizeof(int64_t) * size);
  for (int64_t i = 0; i < entries_size; i++) {
    entries_[i].write(fs_);
  }

  fs_.flush();
}

void BlobDataset::readIndex() {
  entries_.clear();

  fs_.seekg(0, std::ios_base::end);
  if (fs_.tellg() == 0) {
    // skip magic number and index location
    indexOffset_ = 2 * sizeof(int64_t);
    return;
  }

  fs_.seekg(0, std::ios_base::beg);
  int64_t magicNumberCheck = 0;
  fs_.read((char*)&magicNumberCheck, sizeof(int64_t));
  if (magicNumber != magicNumberCheck) {
    throw af::exception(
        "File is not a fl::BlobDataset", __FILE__, __LINE__, AF_ERR_RUNTIME);
  }
  fs_.read((char*)&indexOffset_, sizeof(int64_t));
  fs_.seekg(indexOffset_, std::ios_base::beg);

  int64_t size;
  int64_t entries_size;
  fs_.read((char*)&size, sizeof(int64_t));
  fs_.read((char*)&entries_size, sizeof(int64_t));
  sizes_.resize(size);
  offsets_.resize(size);
  entries_.resize(entries_size);

  fs_.read((char*)sizes_.data(), sizeof(int64_t) * size);
  fs_.read((char*)offsets_.data(), sizeof(int64_t) * size);
  for (int64_t i = 0; i < entries_size; i++) {
    entries_[i].read(fs_);
  }
}

} // namespace fl
