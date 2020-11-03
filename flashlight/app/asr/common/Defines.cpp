/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/common/Defines.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>

namespace fl {
namespace app {
namespace asr {

namespace detail {

/***************************** Deprecated Flags  *****************************/
namespace {

void registerDeprecatedFlags() {
  // Register deprecated flags here using DEPRECATE_FLAGS. For example:
  // DEPRECATE_FLAGS(my_now_deprecated_flag_name, my_new_flag_name);
}

} // namespace

DeprecatedFlagsMap& getDeprecatedFlags() {
  static DeprecatedFlagsMap flagsMap = DeprecatedFlagsMap();
  return flagsMap;
}

void addDeprecatedFlag(
    const std::string& deprecatedFlagName,
    const std::string& newFlagName) {
  auto& map = getDeprecatedFlags();
  map.emplace(deprecatedFlagName, newFlagName);
}

bool isFlagSet(const std::string& name) {
  gflags::CommandLineFlagInfo flagInfo;
  if (!gflags::GetCommandLineFlagInfo(name.c_str(), &flagInfo)) {
    std::stringstream ss;
    ss << "Flag name " << name << " not found - check that it's declared.";
    throw std::invalid_argument(ss.str());
  }
  return !flagInfo.is_default;
}

} // namespace detail

void handleDeprecatedFlags() {
  auto& map = detail::getDeprecatedFlags();
  // Register flags
  static std::once_flag registerFlagsOnceFlag;
  std::call_once(registerFlagsOnceFlag, detail::registerDeprecatedFlags);

  for (auto& flagPair : map) {
    std::string deprecatedFlagValue;
    gflags::GetCommandLineOption(flagPair.first.c_str(), &deprecatedFlagValue);

    bool deprecatedFlagSet = detail::isFlagSet(flagPair.first);
    bool newFlagSet = detail::isFlagSet(flagPair.second);

    if (deprecatedFlagSet && newFlagSet) {
      // Use the new flag value
      std::cerr << "[WARNING] Both deprecated flag " << flagPair.first
                << " and new flag " << flagPair.second
                << " are set. Only the new flag will be "
                << "serialized when the model saved." << std::endl;
      ;
    } else if (deprecatedFlagSet && !newFlagSet) {
      std::cerr
          << "[WARNING] Usage of flag --" << flagPair.first
          << " is deprecated and has been replaced with "
          << "--" << flagPair.second
          << ". Setting the new flag equal to the value of the deprecated flag."
          << "The old flag will not be serialized when the model is saved."
          << std::endl;
      if (gflags::SetCommandLineOption(
              flagPair.second.c_str(), deprecatedFlagValue.c_str())
              .empty()) {
        std::stringstream ss;
        ss << "Failed to set new flag " << flagPair.second << " to value from "
           << flagPair.first << ".";
        throw std::logic_error(ss.str());
      }
    }

    // If the user set the new flag but not the deprecated flag, noop. If the
    // user set neither flag, noop.
  }
}
} // namespace asr
} // namespace app
} // namespace fl
