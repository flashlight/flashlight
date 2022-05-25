/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <iomanip>
#include <sstream>

#include "flashlight/pkg/runtime/Runtime.h"

namespace fl {
namespace pkg {
namespace runtime {

constexpr size_t kRunFileNameIntWidth = 3;

std::string
getRunFile(const std::string& name, const int runidx, const fs::path& runpath) {
  std::stringstream ss;
  ss << std::setw(kRunFileNameIntWidth) << std::setfill('0') << runidx << "_"
     << name;
  return runpath / ss.str();
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
    std::shared_ptr<fl::pkg::runtime::DynamicScaler> dynamicScaler,
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

} // end namespace runtime
} // end namespace pkg
} // end namespace fl
