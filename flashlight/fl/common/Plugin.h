/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include "flashlight/fl/common/Defines.h"

namespace fl {

class FL_API Plugin {
 public:
  explicit Plugin(const std::string& name);
  ~Plugin();

 protected:
  template <typename T>
  T getSymbol(const std::string& symbol) {
    return (T)getRawSymbol(symbol);
  }

 private:
  void* getRawSymbol(const std::string& symbol);
  std::string name_;
  void* handle_;
};
} // namespace fl
