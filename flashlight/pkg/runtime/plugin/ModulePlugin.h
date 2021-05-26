/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Plugin.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {
namespace pkg {
namespace runtime {

typedef Module* (*w2l_module_plugin_t)(int64_t nFeatures, int64_t nClasses);

class ModulePlugin : public Plugin {
 public:
  explicit ModulePlugin(const std::string& name);
  std::shared_ptr<fl::Module> arch(int64_t nFeatures, int64_t nClasses);

 private:
  w2l_module_plugin_t arch_;
};

} // namespace runtime
} // namespace pkg
} // namespace fl
