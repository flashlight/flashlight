/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/flashlight.h"

namespace fl {
namespace pkg {
namespace runtime {

struct Serializer {
 public:
  template <class... Args>
  static void save(
      const fs::path& filepath,
      const std::string& version,
      const Args&... args) {
    fl::retryWithBackoff(
        std::chrono::seconds(1),
        2.0,
        6,
        saveImpl<Args...>,
        filepath,
        version,
        args...); // max wait 31s
  }

  template <typename... Args>
  static void load(const fs::path& filepath, Args&... args) {
    fl::retryWithBackoff(
        std::chrono::seconds(1),
        2.0,
        6,
        loadImpl<Args...>,
        filepath,
        args...); // max wait 31s
  }

 private:
  template <typename... Args>
  static void saveImpl(
      const fs::path& filepath,
      const std::string& version,
      const Args&... args) {
    try {
      std::ofstream file(filepath, std::ios::binary);
      if (!file.is_open()) {
        throw std::runtime_error(
            "failed to open file for writing: " + filepath.string());
      }
      cereal::BinaryOutputArchive ar(file);
      ar(version);
      ar(args...);
    } catch (const std::exception& ex) {
      FL_LOG(fl::LogLevel::ERROR)
          << "Error while saving \"" << filepath << "\": " << ex.what() << "\n";
      throw;
    }
  }

  template <typename... Args>
  static void loadImpl(const fs::path& filepath, Args&... args) {
    try {
      std::ifstream file(filepath, std::ios::binary);
      if (!file.is_open()) {
        throw std::runtime_error(
            "failed to open file for reading: " + filepath.string());
      }
      cereal::BinaryInputArchive ar(file);
      ar(args...);
    } catch (const std::exception& ex) {
      FL_LOG(fl::LogLevel::ERROR) << "Error while loading \"" << filepath
                                  << "\": " << ex.what() << "\n";
      throw;
    }
  }
};
} // namespace runtime
} // namespace pkg
} // namespace fl
