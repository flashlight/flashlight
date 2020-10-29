/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace fl {

namespace detail {

// Inspired from
// https://github.com/facebookincubator/gloo/blob/master/gloo/rendezvous/file_store.h
class FileStore {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(60 * 2);
  explicit FileStore(const std::string& path) : basePath_(path) {}
  std::vector<char> get(const std::string& key);
  void set(const std::string& key, const std::vector<char>& data);
  void clear(const std::string& key);

 private:
  std::string basePath_;

  void wait(const std::string& key);
  bool check(const std::string& key);
  std::string objectPath(const std::string& name);
  std::string tmpPath(const std::string& name);
};
} // namespace detail

} // namespace fl
