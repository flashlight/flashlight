/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "flashlight/fl/flashlight.h"
#include "flashlight/lib/common/System.h"

namespace fl {
namespace ext {

struct Serializer {
 public:
  template <class... Args>
  static void save(
      const std::string& filepath,
      const std::string& version,
      const Args&... args) {
    lib::retryWithBackoff(
        std::chrono::seconds(1),
        2.0,
        6,
        saveImpl<Args...>,
        filepath,
        version,
        args...); // max wait 31s
  }

  template <typename... Args>
  static void load(const std::string& filepath, Args&... args) {
    lib::retryWithBackoff(
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
      const std::string& filepath,
      const std::string& version,
      const Args&... args) {
    try {
      std::ofstream file(filepath, std::ios::binary);
      if (!file.is_open()) {
        throw std::runtime_error(
            "failed to open file for writing: " + filepath);
      }
      cereal::BinaryOutputArchive ar(file);
      ar(version);
      ar(args...);
    } catch (const std::exception& ex) {
      FL_LOG(fl::ERROR) << "Error while saving \"" << filepath
                        << "\": " << ex.what() << "\n";
      throw;
    }
  }

  template <typename... Args>
  static void loadImpl(const std::string& filepath, Args&... args) {
    try {
      std::ifstream file(filepath, std::ios::binary);
      if (!file.is_open()) {
        throw std::runtime_error(
            "failed to open file for reading: " + filepath);
      }
      cereal::BinaryInputArchive ar(file);
      ar(args...);
    } catch (const std::exception& ex) {
      FL_LOG(fl::ERROR) << "Error while loading \"" << filepath
                        << "\": " << ex.what() << "\n";
      throw;
    }
  }
};
} // namespace ext
} // namespace fl
