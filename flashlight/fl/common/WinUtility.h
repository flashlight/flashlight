/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#ifdef _WIN32

namespace fl {
namespace detail {

/**
 * Convert a UTF-8 string to a UTF-16LE wide string for Windows APIs
 * @param[in] utf8 UTF-8 encoded string
 * @return Wide string (UTF-16LE)
 * @throws std::runtime_error if conversion fails
 */
std::wstring utf8ToWide(const std::string& utf8);

/**
 * Get a human-readable error message from the last Windows error code
 * @return Error message as UTF-8 string
 */
std::string getWindowsErrorString();

} // namespace detail
} // namespace fl

#endif // _WIN32
