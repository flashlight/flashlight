/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <vector>

#include <flashlight/flashlight.h>

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
 * Compute the total number of parameters of a fl::Module.
 */
int64_t numTotalParams(std::shared_ptr<fl::Module> module);

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

af::array allreduceGet(AverageValueMeter& mtr);
af::array allreduceGet(EditDistanceMeter& mtr);
af::array allreduceGet(CountMeter& mtr);
af::array allreduceGet(TimeMeter& mtr);

void allreduceSet(AverageValueMeter& mtr, af::array& val);
void allreduceSet(EditDistanceMeter& mtr, af::array& val);
void allreduceSet(CountMeter& mtr, af::array& val);
void allreduceSet(TimeMeter& mtr, af::array& val);

} // namespace ext
} // namespace fl

#include "extensions/common/Utils-inl.h"