/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace ext {

/**
 * Convert an arrayfire array into a std::vector.
 *
 * @param arr input array to convert
 *
 */
template <typename T>
std::vector<T> afToVector(const af::array& arr);

/**
 * Convert the array in a Variable into a std::vector.
 *
 * @param var input Variables to convert
 *
 */
template <typename T>
std::vector<T> afToVector(const fl::Variable& var);

/**
 * Call Flashlight API to initialize distributed environment.
 */
void initDistributed(
    int worldRank,
    int worldSize,
    int maxDevicesPerNode,
    const std::string& rndvFilepath);

/**
 * Synchronize meters across process.
 */
template <typename T>
void syncMeter(T& mtr);

template void syncMeter<AverageValueMeter>(AverageValueMeter& mtr);
template void syncMeter<EditDistanceMeter>(EditDistanceMeter& mtr);
template void syncMeter<CountMeter>(CountMeter& mtr);
template void syncMeter<TimeMeter>(TimeMeter& mtr);
template void syncMeter<TopKMeter>(TopKMeter& mtr);

af::array allreduceGet(AverageValueMeter& mtr);
af::array allreduceGet(EditDistanceMeter& mtr);
af::array allreduceGet(CountMeter& mtr);
af::array allreduceGet(TimeMeter& mtr);
af::array allreduceGet(TopKMeter& mtr);

void allreduceSet(AverageValueMeter& mtr, af::array& val);
void allreduceSet(EditDistanceMeter& mtr, af::array& val);
void allreduceSet(CountMeter& mtr, af::array& val);
void allreduceSet(TimeMeter& mtr, af::array& val);
void allreduceSet(TopKMeter& mtr, af::array& val);

} // namespace ext
} // namespace fl

#include "flashlight/ext/common/Utils-inl.h"
