/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/FileStore.h"

#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace {
static std::string encodeName(const std::string& name) {
  static std::hash<std::string> hashFn;
  return std::to_string(hashFn(name));
}

std::string pathsConcat(const std::string& p1, const std::string& p2) {
  char sep = '/';

#ifdef _WIN32
  sep = '\\';
#endif

  if (!p1.empty() && p1[p1.length() - 1] != sep) {
    return p1 + sep + p2; // Need to add a path separator
  } else {
    return p1 + p2;
  }
}
} // namespace

namespace fl {

namespace detail {

constexpr std::chrono::milliseconds FileStore::kDefaultTimeout;

void FileStore::set(const std::string& key, const std::vector<char>& data) {
  auto tmp = tmpPath(key);
  auto path = objectPath(key);

  {
    // Fail if the key already exists. This implementation is not race free.
    // A race free solution would need to atomically create the file 'path'
    // using an API that fails if the file exists (not provided by STL). If
    // created successfully, rename the temp file as below.
    std::ifstream ifs(path.c_str());
    if (ifs.is_open()) {
      throw std::runtime_error("FileStore set: file already exists: " + path);
    }
  }

  {
    std::ofstream ofs(tmp.c_str(), std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
      throw std::runtime_error("FileStore set: file create failed: " + tmp);
    }
    ofs.write(data.data(), data.size());
  }

  // Atomically move result to final location
  auto rv = rename(tmp.c_str(), path.c_str());
  if (rv != 0) {
    throw std::runtime_error("FileStore set: rename failed");
  }
}

std::vector<char> FileStore::get(const std::string& key) {
  auto path = objectPath(key);
  std::vector<char> result;

  // Block until key is set
  wait(key);

  std::ifstream ifs(path.c_str(), std::ios::in);
  if (!ifs) {
    throw std::runtime_error("FileStore get: file open failed: " + path);
  }

  ifs.seekg(0, std::ios::end);
  size_t n = ifs.tellg();
  if (n == 0) {
    throw std::runtime_error("FileStore get: file is empty: " + path);
  }
  result.resize(n);
  ifs.seekg(0);
  ifs.read(result.data(), n);
  return result;
}

void FileStore::clear(const std::string& key) {
  auto path = objectPath(key);
  remove(path.c_str());
}

bool FileStore::check(const std::string& key) {
  auto path = objectPath(key);

  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    // Only deal with files that don't exist.
    // Anything else is a problem.
    if (errno != ENOENT) {
      throw std::runtime_error("FileStore check: file open failed: " + path);
    }

    // path doesn't exist; return early
    return false;
  }
  close(fd);
  return true;
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

std::string FileStore::tmpPath(const std::string& name) {
  return pathsConcat(basePath_, "." + encodeName(name));
}

std::string FileStore::objectPath(const std::string& name) {
  return pathsConcat(basePath_, encodeName(name));
}

} // namespace detail

} // namespace fl
