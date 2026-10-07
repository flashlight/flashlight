/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/Plugin.h"

#include <sstream>
#include <stdexcept>

#ifdef _WIN32
  #include <windows.h>
  #include "flashlight/fl/common/WinUtility.h"
  #define PLUGIN_HANDLE HMODULE
#else
  #include <dlfcn.h>
  #define PLUGIN_HANDLE void*
#endif

namespace fl {

Plugin::Plugin(const std::string& name) : name_(name) {
#ifdef _WIN32
  auto wideName = detail::utf8ToWide(name);
  handle_ = (void*)LoadLibraryW(wideName.c_str());
  if (!handle_) {
    auto err = detail::getWindowsErrorString();
    throw std::runtime_error("unable to load library <" + name + ">: " + err);
  }
#else
  dlerror(); // clear errors
  handle_ = dlopen(name.c_str(), RTLD_LAZY);
  if (!handle_) {
    auto err = dlerror();
    throw std::runtime_error("unable to load library <" + name + ">: " + err);
  }
#endif
}

void* Plugin::getRawSymbol(const std::string& symbol) {
#ifdef _WIN32
  auto addr = (void*)GetProcAddress((HMODULE)handle_, symbol.c_str());
#else
  dlerror(); // clear errors
  auto addr = dlsym(handle_, symbol.c_str());
#endif

  if (!addr) {
#ifdef _WIN32
    auto err = detail::getWindowsErrorString();
#else
    auto err = dlerror();
#endif
    std::stringstream msg;
    msg << "unable to resolve symbol <" << symbol << ">";
    msg << " in library <" << name_ << ">";
    msg << ":" << err;
    throw std::runtime_error(msg.str());
  }
  return addr;
}

Plugin::~Plugin() {
  if (handle_) {
#ifdef _WIN32
    FreeLibrary((HMODULE)handle_);
#else
    dlclose(handle_);
#endif
  }
}

} // namespace fl
