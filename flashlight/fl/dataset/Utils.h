/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/Dataset.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 * \defgroup dataset_utils Dataset Utils
 * @{
 */

/**
 * Partitions the samples in a round-robin manner and return ids of the samples.
 * For dealing with end effects, we include final samples iff we can fit
 * atleast one sample for last batch for all partitions
 * @param numSamples total number of samples
 * @param partitionId rank of the current partition [0, numPartitions)
 * @param numPartitions total partitions
 * @param batchSz batchsize to be used
 */
FL_API std::vector<int64_t> partitionByRoundRobin(
    int64_t numSamples,
    int64_t partitionId,
    int64_t numPartitions,
    int64_t batchSz = 1,
    bool allowEmpty = false);

/**
 * Partitions the samples in a round-robin manner and return ids of the samples
 * with dynamic batching: max number of tokens in the batch (including padded
 * tokens) should be maxTokens.
 * @param samplesSize samples length in tokens
 * @param partitionId rank of the current partition [0, numPartitions)
 * @param numPartitions total partitions
 * @param maxTokens total number of tokens in the batch
 */
FL_API std::pair<std::vector<int64_t>, std::vector<int64_t>>
dynamicPartitionByRoundRobin(
    const std::vector<float>& samplesSize,
    int64_t partitionId,
    int64_t numPartitions,
    int64_t maxSizePerBatch,
    bool allowEmpty = false);

/**
 * Make batch by applying batchFn to the data
 * @param data data to be batchified
 * @param batchFn function which is applied to make a batch
 */
FL_API Tensor makeBatch(
    const std::vector<Tensor>& data,
    const Dataset::BatchFunction& batchFn = {});

/**
 * Make batch from part of indices (range [start, end) )
 * by applying set of batch functions
 * @param data dataset from which we take particular samples
 * @param batchFns set of functions which are applied to make a batch
 * @param start start index
 * @param end end index
 */
FL_API std::vector<Tensor> makeBatchFromRange(
    std::shared_ptr<const Dataset> dataset,
    std::vector<Dataset::BatchFunction> batchFns,
    int64_t start,
    int64_t end);

/** @} */

} // namespace fl
