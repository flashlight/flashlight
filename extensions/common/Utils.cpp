/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "extensions/common/Utils.h"

#include <flashlight/flashlight.h>

namespace fl {
namespace ext {

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

af::array allreduceGet(fl::AverageValueMeter& mtr) {
  auto mtrVal = mtr.value();
  mtrVal[0] *= mtrVal[2];
  return af::array(mtrVal.size(), mtrVal.data());
}

af::array allreduceGet(fl::EditDistanceMeter& mtr) {
  auto mtrVal = mtr.value();
  mtrVal[0] = mtrVal[0] * mtrVal[1] / 100;
  mtrVal[2] = mtrVal[2] * mtrVal[1] / 100;
  mtrVal[3] = mtrVal[3] * mtrVal[1] / 100;
  mtrVal[4] = mtrVal[4] * mtrVal[1] / 100;

  return af::array(mtrVal.size(), mtrVal.data());
}

af::array allreduceGet(fl::CountMeter& mtr) {
  auto mtrVal0 = mtr.value();
  std::vector<long long> mtrVal(mtrVal0.begin(), mtrVal0.end());
  return af::array(mtrVal.size(), mtrVal.data());
}

af::array allreduceGet(fl::TimeMeter& mtr) {
  return af::constant(mtr.value(), 1, af::dtype::f64);
}

void allreduceSet(fl::AverageValueMeter& mtr, af::array& val) {
  mtr.reset();
  auto valVec = afToVector<double>(val);
  if (valVec[2] != 0) {
    valVec[0] /= valVec[2];
  }
  mtr.add(valVec[0], valVec[2]);
}

void allreduceSet(fl::EditDistanceMeter& mtr, af::array& val) {
  mtr.reset();
  auto valVec = afToVector<double>(val);
  mtr.add(
      static_cast<int64_t>(valVec[1]),
      static_cast<int64_t>(valVec[2]),
      static_cast<int64_t>(valVec[3]),
      static_cast<int64_t>(valVec[4]));
}

void allreduceSet(fl::CountMeter& mtr, af::array& val) {
  mtr.reset();
  auto valVec = afToVector<long long>(val);
  for (size_t i = 0; i < valVec.size(); ++i) {
    mtr.add(i, valVec[i]);
  }
}

void allreduceSet(fl::TimeMeter& mtr, af::array& val) {
  auto worldSize = fl::getWorldSize();
  auto valVec = afToVector<double>(val);
  mtr.set(valVec[0] / worldSize);
}
} // namespace ext
} // namespace fl