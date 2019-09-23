/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <arrayfire.h>

#define AF_CHECK(fn)                                                          \
  do {                                                                        \
    af_err __err = fn;                                                        \
    if (__err == AF_SUCCESS) {                                                \
      break;                                                                  \
    }                                                                         \
    throw af::exception(                                                      \
        "ArrayFire error: ", __PRETTY_FUNCTION__, __FILE__, __LINE__, __err); \
  } while (0)

namespace fl {

// This is a temporary solution for a bug that is fixed in gcc 6.1+. After
// upgrade to that version+, this can be removed. Read more:
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60970

struct EnumNaiveHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

template <typename T>
using u_set = std::unordered_set<T, EnumNaiveHash>;

template <typename KeyType, typename ValueType>
using u_map = std::unordered_map<KeyType, ValueType, EnumNaiveHash>;

/**
 * Returns true if two arrays are of same type and are element-wise equal within
 * given tolerance limit.
 *
 * @param [a,b] input arrays to compare
 * @param absTolerance absolute tolerance allowed
 */
bool allClose(
    const af::array& a,
    const af::array& b,
    double absTolerance = 1e-5);

// Returns high resolution time formatted as:
// MMDD HH MM SS UUUUUU
// 0206 08:42:42.123456
std::string dateTimeWithMicroSeconds();

// Returns round-up result of integer division.
// throws invalid_argument exception on zero denominator.
size_t divRoundUp(size_t numerator, size_t denominator);
/*
 * Converts string to arrayfire types (`af::dtype`).
 *
 * @param[in] typeName type name in string.
 *
 * @return returns an arrayfire type (`af::dtype`) according to the input
 * string.
 */
af::dtype stringToAfType(const std::string& typeName);

/*
 * Converts arrayfire types to human readable string.
 *
 * @param[in] type `af::dtype` whose name in `string` is required.
 *
 * @return returns the type name in string
 */
std::string afTypeToString(const af::dtype& type);

} // namespace fl
