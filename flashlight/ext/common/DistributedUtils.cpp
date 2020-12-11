/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/common/DistributedUtils.h"

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace ext {

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
  auto mtrVal0 = mtr.value();
  std::vector<long long> mtrVal(mtrVal0.begin(), mtrVal0.end());
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

af::array allreduceGet(fl::TopKMeter& mtr) {
  std::pair<int32_t, int32_t> stats = mtr.getStats();
  std::vector<int32_t> vec = {stats.first, stats.second};
  return af::array(vec.size(), vec.data());
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
  auto valVec = afToVector<long long>(val);
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

void allreduceSet(fl::TopKMeter& mtr, af::array& val) {
  mtr.reset();
  auto valVec = afToVector<int32_t>(val);
  mtr.set(valVec[0], valVec[1]);
}
} // namespace ext
} // namespace fl
