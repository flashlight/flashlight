/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

#include "flashlight/fl/distributed/DistributedApi.h"
#include "flashlight/fl/meter/meters.h"

namespace fl {
class Tensor;

namespace pkg {
namespace runtime {

/**
 * Call Flashlight API to initialize distributed environment.
 */
void initDistributed(
    int worldRank,
    int worldSize,
    int maxDevicesPerNode,
    const std::string& rndvFilepath);

Tensor allreduceGet(AverageValueMeter& mtr);
Tensor allreduceGet(EditDistanceMeter& mtr);
Tensor allreduceGet(CountMeter& mtr);
Tensor allreduceGet(TimeMeter& mtr);
Tensor allreduceGet(TopKMeter& mtr);

void allreduceSet(AverageValueMeter& mtr, Tensor& val);
void allreduceSet(EditDistanceMeter& mtr, Tensor& val);
void allreduceSet(CountMeter& mtr, Tensor& val);
void allreduceSet(TimeMeter& mtr, Tensor& val);
void allreduceSet(TopKMeter& mtr, Tensor& val);

/**
 * Synchronize meters across process.
 */
template <typename T>
void syncMeter(T& mtr) {
  if (!fl::isDistributedInit()) {
    return;
  }
  Tensor arr = allreduceGet(mtr);
  fl::allReduce(arr);
  allreduceSet(mtr, arr);
}

template void syncMeter<AverageValueMeter>(AverageValueMeter& mtr);
template void syncMeter<EditDistanceMeter>(EditDistanceMeter& mtr);
template void syncMeter<CountMeter>(CountMeter& mtr);
template void syncMeter<TimeMeter>(TimeMeter& mtr);
template void syncMeter<TopKMeter>(TopKMeter& mtr);

} // namespace runtime
} // namespace pkg
} // namespace fl
