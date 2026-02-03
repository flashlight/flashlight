/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/WinUtility.h"

#ifdef _WIN32

#include <windows.h>
#include <stdexcept>

namespace fl {
namespace detail {

std::wstring utf8ToWide(const std::string& utf8) {
  if (utf8.empty()) {
    return std::wstring();
  }

  int wideSize = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
  if (wideSize == 0) {
    throw std::runtime_error("Failed to convert UTF-8 to wide string");
  }

  std::wstring wide(wideSize - 1, 0);
  MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wide[0], wideSize);
  return wide;
}

std::string getWindowsErrorString() {
  DWORD error = GetLastError();
  if (error == 0) {
    return "No error";
  }

  LPWSTR messageBuffer = nullptr;
  FormatMessageW(
      FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
          FORMAT_MESSAGE_IGNORE_INSERTS,
      nullptr,
      error,
      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
      (LPWSTR)&messageBuffer,
      0,
      nullptr);

  std::string result;
  if (messageBuffer) {
    int utf8Size = WideCharToMultiByte(CP_UTF8, 0, messageBuffer, -1, nullptr,
                                       0, nullptr, nullptr);
    if (utf8Size > 0) {
      result.resize(utf8Size - 1);
      WideCharToMultiByte(CP_UTF8, 0, messageBuffer, -1, &result[0], utf8Size,
                          nullptr, nullptr);
    }
    LocalFree(messageBuffer);
  } else {
    result = "Unknown error";
  }
  return result;
}

} // namespace detail
} // namespace fl

#endif // _WIN32
