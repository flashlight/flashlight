/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>
#include <sstream>

#include "flashlight/pkg/runtime/Runtime.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

namespace fl {
namespace app {

using fl::lib::format;
using fl::lib::pathsConcat;

std::string
getRunFile(const std::string& name, int runidx, const std::string& runpath) {
  auto fname = format("%03d_%s", runidx, name.c_str());
  return pathsConcat(runpath, fname);
};

std::string serializeGflags(const std::string& separator) {
  std::stringstream serialized;
  std::vector<gflags::CommandLineFlagInfo> allFlags;
  gflags::GetAllFlags(&allFlags);
  std::string currVal;
  for (auto itr = allFlags.begin(); itr != allFlags.end(); ++itr) {
    gflags::GetCommandLineOption(itr->name.c_str(), &currVal);
    serialized << "--" << itr->name << "=" << currVal << separator;
  }
  return serialized.str();
}

bool backwardWithScaling(
    const fl::Variable& loss,
    std::vector<fl::Variable>& params,
    std::shared_ptr<fl::ext::DynamicScaler> dynamicScaler,
    std::shared_ptr<fl::Reducer> reducer) {
  auto scaledLoss = loss;
  if (dynamicScaler) {
    scaledLoss = dynamicScaler->scale(loss);
  }

  scaledLoss.backward();
  if (reducer) {
    reducer->finalize();
  }

  if (dynamicScaler) {
    if (!dynamicScaler->unscale(params)) {
      return false;
    }
    dynamicScaler->update();
  }

  return true;
}

} // end namespace app
} // end namespace fl
