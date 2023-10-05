/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/fl/dataset/FileBlobDataset.h"

namespace fl {

FileBlobDataset::FileBlobDataset(
    const fs::path& name,
    bool rw,
    bool truncate)
    : name_(name) {
  mode_ = (rw ? std::ios_base::in | std::ios_base::out : std::ios_base::in);
  {
    std::ofstream fs(name_, (truncate ? mode_ | std::ios_base::trunc : mode_));
    if (!fs.is_open()) {
      throw std::runtime_error("could not open file " + name.string());
    }
  }
  readIndex();
}

std::shared_ptr<std::fstream> FileBlobDataset::getStream() const {
  static thread_local std::shared_ptr<
      std::unordered_map<uintptr_t, std::shared_ptr<std::fstream>>>
      threadFileHandles = std::make_shared<
          std::unordered_map<uintptr_t, std::shared_ptr<std::fstream>>>();

  // Get a per-thread file handle.
  auto keyval = threadFileHandles->find(reinterpret_cast<uintptr_t>(this));
  if (keyval == threadFileHandles->end()) {
    auto fs = std::make_shared<std::fstream>();
    fs->exceptions(
        std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
    fs->open(name_, mode_);
    threadFileHandles->insert({reinterpret_cast<uintptr_t>(this), fs});
    // Link threadFileHandles to the object
    // so the file handle can be cleaned at destruction.
    {
      std::lock_guard<std::mutex> lock(afhmutex_);
      auto i = allFileHandles_.begin();
      bool match = false;
      while (i != std::end(allFileHandles_)) {
        auto ptr = i->lock();
        if (ptr) {
          if (threadFileHandles == ptr) {
            match = true;
          }
          ++i;
        } else {
          i = allFileHandles_.erase(i);
        }
      }
      if (!match) {
        allFileHandles_.push_back(threadFileHandles);
      }
    }
    return fs;
  } else {
    return keyval->second;
  }
}

int64_t FileBlobDataset::writeData(
    int64_t offset,
    const char* data,
    int64_t size) const {
  auto fs = getStream();
  fs->seekp(offset, std::ios_base::beg);
  fs->write(data, size);
  return fs->tellp() - offset;
}

int64_t FileBlobDataset::readData(int64_t offset, char* data, int64_t size)
    const {
  auto fs = getStream();
  fs->seekg(offset, std::ios_base::beg);
  fs->read(data, size);
  return fs->tellg() - offset;
}

void FileBlobDataset::flushData() {
  auto fs = getStream();
  fs->flush();
}

bool FileBlobDataset::isEmptyData() const {
  auto fs = getStream();
  fs->seekg(0, std::ios_base::end);
  return (fs->tellg() == 0);
}

FileBlobDataset::~FileBlobDataset() {
  std::lock_guard<std::mutex> lock(afhmutex_);
  for (auto& weakFileHandles : allFileHandles_) {
    auto fileHandles = weakFileHandles.lock();
    if (fileHandles) {
      fileHandles->erase(reinterpret_cast<uintptr_t>(this));
    }
  }
}

} // namespace fl
