/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/Plugin.h"

#include <dlfcn.h>
#include <sstream>
#include <stdexcept>

namespace fl {

Plugin::Plugin(const std::string& name) : name_(name) {
  dlerror(); // clear errors
  handle_ = dlopen(name.c_str(), RTLD_LAZY);
  if (!handle_) {
    std::string err = dlerror();
    throw std::runtime_error("unable to load library <" + name + ">: " + err);
  }
}

void* Plugin::getRawSymbol(const std::string& symbol) {
  dlerror(); // clear errors
  auto addr = dlsym(handle_, symbol.c_str());
  if (!addr) {
    std::string err = dlerror();
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
    dlclose(handle_);
  }
}

} // namespace fl
