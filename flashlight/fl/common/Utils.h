/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

namespace fl {

class Tensor;

/**
 * @return if fp16 operations are supported with the current flashlight
 * configuration.
 */
bool f16Supported();

// Returns high resolution time formatted as:
// MMDD HH MM SS UUUUUU
// 0206 08:42:42.123456
std::string dateTimeWithMicroSeconds();

// Returns round-up result of integer division.
// throws invalid_argument exception on zero denominator.
size_t divRoundUp(size_t numerator, size_t denominator);

// Return a string formmated similar to: 1314127872(1GB+229MB+256KB)
std::string prettyStringMemorySize(size_t size);

// Returns a string formatted similar to: 26675644(2m+667k+5644)
std::string prettyStringCount(size_t count);

/** @} */

} // namespace fl
