/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <iomanip>
#include <sstream>

#include "flashlight/pkg/runtime/Runtime.h"

namespace fl::pkg::runtime {

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

std::string getCurrentDate() {
  time_t now = time(nullptr);
  struct tm tmbuf;
  struct tm* tstruct;
  tstruct = localtime_r(&now, &tmbuf);

  std::array<char, 80> buf;
  strftime(buf.data(), buf.size(), "%Y-%m-%d", tstruct);
  return std::string(buf.data());
}

std::string getCurrentTime() {
  time_t now = time(nullptr);
  struct tm tmbuf;
  struct tm* tstruct;
  tstruct = localtime_r(&now, &tmbuf);

  std::array<char, 80> buf;
  strftime(buf.data(), buf.size(), "%X", tstruct);
  return std::string(buf.data());
}

} // end namespace fl
