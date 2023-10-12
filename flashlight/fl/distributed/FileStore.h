/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/Filesystem.h"

namespace fl {

namespace detail {

// Inspired from
// https://github.com/facebookincubator/gloo/blob/master/gloo/rendezvous/file_store.h
class FL_API FileStore {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(60 * 2);
  explicit FileStore(const fs::path& path) : basePath_(path) {}
  std::vector<char> get(const std::string& key);
  void set(const std::string& key, const std::vector<char>& data);
  void clear(const std::string& key);

 private:
  fs::path basePath_;

  void wait(const std::string& key);
  bool check(const std::string& key);
  fs::path objectPath(const std::string& name);
  fs::path tmpPath(const std::string& name);
};
} // namespace detail

} // namespace fl
