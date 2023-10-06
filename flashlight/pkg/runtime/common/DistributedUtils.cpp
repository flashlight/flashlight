/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/runtime/common/DistributedUtils.h"

#include "flashlight/fl/flashlight.h"

namespace fl::pkg::runtime {

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

Tensor allreduceGet(fl::AverageValueMeter& mtr) {
  auto mtrVal = mtr.value();
  mtrVal[0] *= mtrVal[2];
  return Tensor::fromVector(mtrVal);
}

Tensor allreduceGet(fl::EditDistanceMeter& mtr) {
  auto mtrVal0 = mtr.value();
  std::vector<long long> mtrVal(mtrVal0.begin(), mtrVal0.end());
  return Tensor::fromVector(mtrVal);
}

Tensor allreduceGet(fl::CountMeter& mtr) {
  auto mtrVal0 = mtr.value();
  std::vector<long long> mtrVal(mtrVal0.begin(), mtrVal0.end());
  return Tensor::fromVector(mtrVal);
}

Tensor allreduceGet(fl::TimeMeter& mtr) {
  return fl::full({1}, mtr.value(), fl::dtype::f64);
}

Tensor allreduceGet(fl::TopKMeter& mtr) {
  std::pair<int32_t, int32_t> stats = mtr.getStats();
  std::vector<int32_t> vec = {stats.first, stats.second};
  return Tensor::fromVector(vec);
}

void allreduceSet(fl::AverageValueMeter& mtr, Tensor& val) {
  mtr.reset();
  auto valVec = val.toHostVector<double>();
  if (valVec[2] != 0) {
    valVec[0] /= valVec[2];
  }
  mtr.add(valVec[0], valVec[2]);
}

void allreduceSet(fl::EditDistanceMeter& mtr, Tensor& val) {
  mtr.reset();
  auto valVec = val.toHostVector<long long>();
  mtr.add(
      static_cast<int64_t>(valVec[1]),
      static_cast<int64_t>(valVec[2]),
      static_cast<int64_t>(valVec[3]),
      static_cast<int64_t>(valVec[4]));
}

void allreduceSet(fl::CountMeter& mtr, Tensor& val) {
  mtr.reset();
  auto valVec = val.toHostVector<long long>();
  for (size_t i = 0; i < valVec.size(); ++i) {
    mtr.add(i, valVec[i]);
  }
}

void allreduceSet(fl::TimeMeter& mtr, Tensor& val) {
  auto worldSize = fl::getWorldSize();
  auto valVec = val.toHostVector<double>();
  mtr.set(valVec[0] / worldSize);
}

void allreduceSet(fl::TopKMeter& mtr, Tensor& val) {
  mtr.reset();
  auto valVec = val.toHostVector<int32_t>();
  mtr.set(valVec[0], valVec[1]);
}
} // namespace fl
