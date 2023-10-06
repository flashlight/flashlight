/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/runtime/plugin/ModulePlugin.h"

namespace fl::pkg::runtime {

ModulePlugin::ModulePlugin(const std::string& name) : fl::Plugin(name) {
  arch_ = getSymbol<w2l_module_plugin_t>("createModule");
}

std::shared_ptr<fl::Module> ModulePlugin::arch(
    int64_t nFeatures,
    int64_t nClasses) {
  return std::shared_ptr<fl::Module>(arch_(nFeatures, nClasses));
}

} // namespace fl
