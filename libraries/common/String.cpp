/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/common/String.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <array>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>

static constexpr const char* kSpaceChars = "\t\n\v\f\r ";

namespace fl {
namespace lib {

std::string trim(const std::string& str) {
  auto i = str.find_first_not_of(kSpaceChars);
  if (i == std::string::npos) {
    return "";
  }
  auto j = str.find_last_not_of(kSpaceChars);
  if (j == std::string::npos || i > j) {
    return "";
  }
  return str.substr(i, j - i + 1);
}

void replaceAll(
    std::string& str,
    const std::string& from,
    const std::string& repl) {
  if (from.empty()) {
    return;
  }
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.length(), repl);
    pos += repl.length();
  }
}

bool startsWith(const std::string& input, const std::string& pattern) {
  return (input.find(pattern) == 0);
}

template <bool Any, typename Delim>
static std::vector<std::string> splitImpl(
    const Delim& delim,
    std::string::size_type delimSize,
    const std::string& input,
    bool ignoreEmpty = false) {
  std::vector<std::string> result;
  std::string::size_type i = 0;
  while (true) {
    auto j = Any ? input.find_first_of(delim, i) : input.find(delim, i);
    if (j == std::string::npos) {
      break;
    }
    if (!(ignoreEmpty && i == j)) {
      result.emplace_back(input.begin() + i, input.begin() + j);
    }
    i = j + delimSize;
  }
  if (!(ignoreEmpty && i == input.size())) {
    result.emplace_back(input.begin() + i, input.end());
  }
  return result;
}

std::vector<std::string>
split(char delim, const std::string& input, bool ignoreEmpty) {
  return splitImpl<false>(delim, 1, input, ignoreEmpty);
}

std::vector<std::string>
split(const std::string& delim, const std::string& input, bool ignoreEmpty) {
  if (delim.empty()) {
    throw std::invalid_argument("delimiter is empty string");
  }
  return splitImpl<false>(delim, delim.size(), input, ignoreEmpty);
}

std::vector<std::string> splitOnAnyOf(
    const std::string& delim,
    const std::string& input,
    bool ignoreEmpty) {
  return splitImpl<true>(delim, 1, input, ignoreEmpty);
}

std::vector<std::string> splitOnWhitespace(
    const std::string& input,
    bool ignoreEmpty) {
  return splitOnAnyOf(kSpaceChars, input, ignoreEmpty);
}

std::string join(
    const std::string& delim,
    const std::vector<std::string>& vec) {
  return join(delim, vec.begin(), vec.end());
}

}
}