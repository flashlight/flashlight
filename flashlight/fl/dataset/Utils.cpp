/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/dataset/Utils.h"

#include <algorithm>
#include <stdexcept>

#include "flashlight/fl/tensor/Index.h"

namespace fl {

std::vector<int64_t> partitionByRoundRobin(
    int64_t numSamples,
    int64_t partitionId,
    int64_t numPartitions,
    int64_t batchSz /* = 1 */,
    bool allowEmpty /* = false */) {
  if (partitionId < 0 || partitionId >= numPartitions) {
    throw std::invalid_argument(
        "invalid partitionId, numPartitions for partitionByRoundRobin");
  }
  int64_t nSamplesPerGlobalBatch = numPartitions * batchSz;
  int64_t nGlobalBatches = numSamples / nSamplesPerGlobalBatch;
  bool includeLast = (numSamples % nSamplesPerGlobalBatch) >= numPartitions;
  if (allowEmpty && (numSamples % nSamplesPerGlobalBatch) > 0) {
    includeLast = true;
  }
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

std::pair<std::vector<int64_t>, std::vector<int64_t>>
dynamicPartitionByRoundRobin(
    const std::vector<float>& samplesSize,
    int64_t partitionId,
    int64_t numPartitions,
    int64_t maxSizePerBatch,
    bool allowEmpty /* = false */) {
  if (partitionId < 0 || partitionId >= numPartitions) {
    throw std::invalid_argument(
        "[dynamicPartitionByRoundRobin] invalid partitionId, numPartitions");
  }
  std::vector<int64_t> batchSizes, batchOffsets;
  int64_t sampleIdx = 0, batchStartSampleIdx = 0;
  float maxSampleLen = 0;
  while (sampleIdx < samplesSize.size()) {
    if (samplesSize[sampleIdx] > maxSizePerBatch) {
      throw std::invalid_argument(
          "[dynamicPartitionByRoundRobin] invalid samples length: each sample "
          "should have size <= maxSizePerBatch, either filter data or set larger maxSizePerBatch. "
          "maxSizePerBatch were set to " +
          std::to_string(maxSizePerBatch) + " sample size is " +
          std::to_string(samplesSize[sampleIdx]));
    }
    float maxSampleLenOld = maxSampleLen;
    maxSampleLen = std::max(maxSampleLen, samplesSize[sampleIdx]);
    if ((sampleIdx - batchStartSampleIdx + 1) * maxSampleLen >
        maxSizePerBatch) {
      if (maxSampleLenOld * (sampleIdx - batchStartSampleIdx) >
          maxSizePerBatch) {
        throw std::invalid_argument(
            "dynamicPartitionByRoundRobin is doing wrong packing");
      }
      batchSizes.push_back(sampleIdx - batchStartSampleIdx);
      batchOffsets.push_back(batchStartSampleIdx);
      batchStartSampleIdx = sampleIdx;
      maxSampleLen = samplesSize[sampleIdx];
    } else {
      sampleIdx++;
    }
  }
  // process last batch with sampleIdx == numSamples, batchStartSampleIdx <
  // numSamples
  if ((sampleIdx - batchStartSampleIdx) * maxSampleLen < maxSizePerBatch) {
    batchSizes.push_back(sampleIdx - batchStartSampleIdx);
    batchOffsets.push_back(batchStartSampleIdx);
  }

  int64_t nGlobalBatches = batchSizes.size() / numPartitions;
  if (allowEmpty && (batchSizes.size() % numPartitions) > 0) {
    ++nGlobalBatches;
  }
  std::vector<int64_t> outSamples, outBatchSizes;
  for (size_t i = 0; i < nGlobalBatches; i++) {
    int index = i * numPartitions + partitionId;
    if (index < batchSizes.size()) {
      outBatchSizes.emplace_back(batchSizes[index]);
      for (int64_t b = 0; b < batchSizes[index]; ++b) {
        outSamples.emplace_back(b + batchOffsets[index]);
      }
    }
  }
  return {outSamples, outBatchSizes};
}

std::vector<Tensor> makeBatchFromRange(
    std::shared_ptr<const Dataset> dataset,
    std::vector<Dataset::BatchFunction> batchFns,
    int64_t start,
    int64_t end) {
  std::vector<std::vector<Tensor>> buffer;
  for (int64_t batchidx = start; batchidx < end; ++batchidx) {
    auto fds = dataset->get(batchidx);
    if (buffer.size() < fds.size()) {
      buffer.resize(fds.size());
    }
    for (int64_t i = 0; i < fds.size(); ++i) {
      buffer[i].emplace_back(fds[i]);
    }
  }
  std::vector<Tensor> result(buffer.size());
  for (int64_t i = 0; i < buffer.size(); ++i) {
    result[i] =
        makeBatch(buffer[i], (i < batchFns.size()) ? batchFns[i] : nullptr);
  }
  return result;
}

Tensor makeBatch(
    const std::vector<Tensor>& data,
    const Dataset::BatchFunction& batchFn) {
  if (batchFn) {
    return batchFn(data);
  }
  // Using default batching function
  if (data.empty()) {
    return Tensor();
  }
  auto& dims = data[0].shape();

  for (const auto& d : data) {
    if (d.shape() != dims) {
      throw std::invalid_argument("dimension mismatch while batching dataset");
    }
  }

  int ndims = (data[0].elements() > 1) ? dims.ndim() : 0;

  // TODO: expand this to > 4 given fl::Tensor - should work out of the box
  // by just removing this check? Possibly also change to ndims >= dims.ndims()
  if (ndims >= 4) {
    throw std::invalid_argument("# of dims must be < ndim - 1 for batching");
  }
  // Dimensions of the batched tensor
  std::vector<Dim> batchDims = dims.get();
  if (ndims + 1 > batchDims.size()) {
    batchDims.push_back(1); // placeholder dim
  }
  batchDims[ndims] = data.size();
  auto batcharr = Tensor(Shape(batchDims), data[0].type());

  for (size_t i = 0; i < data.size(); ++i) {
    std::vector<fl::Index> sel(batcharr.ndim(), fl::span);
    sel[ndims] = i;
    batcharr(sel) = data[i];
  }
  return batcharr;
}
} // namespace fl
