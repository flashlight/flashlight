/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/flashlight/dataset/Dataset.h"

namespace fl {

/**
 * Partitions the samples in a round-robin manner and return ids of the samples.
 * For dealing with end effects, we include final samples iff we can fit
 * atleast one sample for last batch for all partitions
 * @param numSamples total number of samples
 * @param partitionId rank of the current partition [0, numPartitions)
 * @param numPartitions total partitions
 * @param batchSz batchsize to be used
 */
std::vector<int64_t> partitionByRoundRobin(
    int64_t numSamples,
    int64_t partitionId,
    int64_t numPartitions,
    int64_t batchSz = 1);

} // namespace fl
