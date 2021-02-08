/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

#include <arrayfire.h>

/**
 * \defgroup common_utils Common utilities
 * @{
 */

#define AF_CHECK(fn)                                                          \
  do {                                                                        \
    af_err __err = fn;                                                        \
    if (__err == AF_SUCCESS) {                                                \
      break;                                                                  \
    }                                                                         \
    throw af::exception(                                                      \
        "ArrayFire error: ", __PRETTY_FUNCTION__, __FILE__, __LINE__, __err); \
  } while (0)

std::ostream& operator<<(std::ostream& os, af::dtype type);

namespace fl {

/**
 * @return if fp16 operations are supported with the current flashlight
 * configuration.
 */
bool f16Supported();

/**
 * Returns true if two arrays are of same type and are element-wise equal within
 * a given tolerance limit.
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

// Return a string formmated similar to: 1314127872(1GB+229MB+256KB)
std::string prettyStringMemorySize(size_t size);

// Returns a string formatted similar to: 26675644(2m+667k+5644)
std::string prettyStringCount(size_t count);

/** @} */

std::string prettyStringCount(af::dtype type);

} // namespace fl
