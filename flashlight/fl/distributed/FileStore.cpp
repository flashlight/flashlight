/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/FileStore.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

static std::string encodeName(const std::string& name) {
  static std::hash<std::string> hashFn;
  return std::to_string(hashFn(name));
}

} // namespace

namespace fl::detail {

constexpr std::chrono::milliseconds FileStore::kDefaultTimeout;

void FileStore::set(const std::string& key, const std::vector<char>& data) {
  fs::path tmp = tmpPath(key);
  fs::path path = objectPath(key);

  {
    // Fail if the key already exists. This implementation is not race free.
    // A race free solution would need to atomically create the file 'path'
    // using an API that fails if the file exists (not provided by STL). If
    // created successfully, rename the temp file as below.
    std::ifstream ifs(path);
    if (ifs.is_open()) {
      throw std::runtime_error(
          "FileStore set: file already exists: " + path.string());
    }
  }

  {
    std::ofstream ofs(tmp, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
      throw std::runtime_error(
          "FileStore set: file create failed: " + tmp.string());
    }
    ofs.write(data.data(), data.size());
  }

  // Atomically move result to final location
  fs::rename(tmp, path);
}

std::vector<char> FileStore::get(const std::string& key) {
  fs::path path = objectPath(key);
  std::vector<char> result;

  // Block until key is set
  wait(key);

  std::ifstream ifs(path, std::ios::in);
  if (!ifs) {
    throw std::runtime_error(
        "FileStore get: file open failed: " + path.string());
  }

  ifs.seekg(0, std::ios::end);
  size_t n = ifs.tellg();
  if (n == 0) {
    throw std::runtime_error("FileStore get: file is empty: " + path.string());
  }
  result.resize(n);
  ifs.seekg(0);
  ifs.read(result.data(), n);
  return result;
}

void FileStore::clear(const std::string& key) {
  fs::path path = objectPath(key);
  fs::remove(path);
}

bool FileStore::check(const std::string& key) {
  fs::path path = objectPath(key);
  return fs::exists(path);
}

void FileStore::wait(const std::string& key) {
  // Not using inotify because it doesn't work on many
  // shared filesystems (such as NFS).
  const auto start = std::chrono::steady_clock::now();
  while (!check(key)) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (elapsed > FileStore::kDefaultTimeout) {
      throw std::runtime_error("FileStore timed out for key: " + key);
    }
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

fs::path FileStore::tmpPath(const std::string& name) {
  return basePath_ / fs::path("." + encodeName(name));
}

fs::path FileStore::objectPath(const std::string& name) {
  return basePath_ / fs::path(encodeName(name));
}

} // namespace fl
