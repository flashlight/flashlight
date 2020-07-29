/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>


namespace w2l {

// ============================ Types and Templates ============================

template <typename It>
using DecayDereference =
    typename std::decay<decltype(*std::declval<It>())>::type;

template <typename S, typename T>
using EnableIfSame = typename std::enable_if<std::is_same<S, T>::value>::type;

// ================================== Functions ==================================

std::string trim(const std::string& str);

void replaceAll(
    std::string& str,
    const std::string& from,
    const std::string& repl);

bool startsWith(const std::string& input, const std::string& pattern);

std::vector<std::string>
split(char delim, const std::string& input, bool ignoreEmpty = false);

std::vector<std::string> split(
    const std::string& delim,
    const std::string& input,
    bool ignoreEmpty = false);

std::vector<std::string> splitOnAnyOf(
    const std::string& delim,
    const std::string& input,
    bool ignoreEmpty = false);

std::vector<std::string> splitOnWhitespace(
    const std::string& input,
    bool ignoreEmpty = false);

/**
 * Join a vector of `std::string` inserting `delim` in between.
 */
std::string join(const std::string& delim, const std::vector<std::string>& vec);

/**
 * Join a range of `std::string` specified by iterators.
 */
template <
    typename FwdIt,
    typename = EnableIfSame<DecayDereference<FwdIt>, std::string>>
std::string join(const std::string& delim, FwdIt begin, FwdIt end) {
      if (begin == end) {
    return "";
  }

  size_t totalSize = begin->size();
  for (auto it = std::next(begin); it != end; ++it) {
    totalSize += delim.size() + it->size();
  }

  std::string result;
  result.reserve(totalSize);

  result.append(*begin);
  for (auto it = std::next(begin); it != end; ++it) {
    result.append(delim);
    result.append(*it);
  }
  return result;
}

/**
 * Create an output string using a `printf`-style format string and arguments.
 * Safer than `sprintf` which is vulnerable to buffer overflow.
 */
template <class... Args>
std::string format(const char* fmt, Args&&... args) {
  auto res = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
  if (res < 0) {
    throw std::runtime_error(std::strerror(errno));
  }
  std::string buf(res, '\0');
  // the size here is fine -- it's legal to write '\0' to buf[res]
  auto res2 = std::snprintf(&buf[0], res + 1, fmt, std::forward<Args>(args)...);
  if (res2 < 0) {
    throw std::runtime_error(std::strerror(errno));
  }

  if (res2 != res) {
    throw std::runtime_error(
        "The size of the formated string is not equal to what it is expected.");
  }
  return buf;
}

/**
 * Dedup the elements in a vector.
 */
template <class T>
void dedup(std::vector<T>& in) {
  if (in.empty()) {
    return;
  }
  auto it = std::unique(in.begin(), in.end());
  in.resize(std::distance(in.begin(), it));
}

}