/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight-extensions/common/Utils.h"

#include <flashlight/flashlight.h>

using namespace fl;

namespace {
std::shared_ptr<Module> parseLine(const std::string& line);

std::shared_ptr<Module> parseLines(
    const std::vector<std::string>& lines,
    const int lineIdx,
    int& numLinesParsed);
} // namespace


namespace w2l {

int64_t numTotalParams(std::shared_ptr<fl::Module> module) {
  int64_t params = 0;
  for (auto& p : module->params()) {
    params += p.elements();
  }
  return params;
}

void initDistributed(
    int worldRank,
    int worldSize,
    int maxDevicesPerNode,
    const std::string& rndvFilepath) {
  if (rndvFilepath.empty()) {
    distributedInit(
        fl::DistributedInit::MPI,
        -1, // unused for MPI
        -1, // unused for MPI
        {{fl::DistributedConstants::kMaxDevicePerNode,
          std::to_string(maxDevicesPerNode)}});
  } else {
    distributedInit(
        fl::DistributedInit::FILE_SYSTEM,
        worldRank,
        worldSize,
        {{fl::DistributedConstants::kMaxDevicePerNode,
          std::to_string(maxDevicesPerNode)},
         {fl::DistributedConstants::kFilePath, rndvFilepath}});
  }
}

}