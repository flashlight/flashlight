/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/flashlight/dataset/Utils.h"

#include <algorithm>
#include <stdexcept>

namespace fl {

std::vector<int64_t> partitionByRoundRobin(
    int64_t numSamples,
    int64_t partitionId,
    int64_t numPartitions,
    int64_t batchSz /* = 1 */) {
  if (partitionId < 0 || partitionId >= numPartitions) {
    throw std::invalid_argument(
        "invalid partitionId, numPartitions for partitionByRoundRobin");
  }
  int64_t nSamplesPerGlobalBatch = numPartitions * batchSz;
  int64_t nGlobalBatches = numSamples / nSamplesPerGlobalBatch;
  bool includeLast = (numSamples % nSamplesPerGlobalBatch) >= numPartitions;
  if (includeLast) {
    ++nGlobalBatches;
  }
  std::vector<int64_t> outSamples;
  outSamples.reserve(nGlobalBatches * batchSz);

  for (size_t i = 0; i < nGlobalBatches; i++) {
    auto offset = i * nSamplesPerGlobalBatch;
    int64_t nCurSamples; // num samples in current batch
    if (includeLast && (i == nGlobalBatches - 1)) {
      nCurSamples =
          (numSamples - offset) / numPartitions; // min samples per proc
      int64_t remaining = (numSamples - offset) % numPartitions;
      offset += nCurSamples * partitionId;
      if (partitionId < remaining) {
        nCurSamples += 1;
      }
      offset += std::min(partitionId, remaining);
    } else {
      offset += batchSz * partitionId;
      nCurSamples = batchSz;
    }
    for (int64_t b = 0; b < nCurSamples; ++b) {
      outSamples.emplace_back(b + offset);
    }
  }
  return outSamples;
}
} // namespace fl
