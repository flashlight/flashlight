/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/dataset/ResampleDataset.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace {

std::vector<int64_t> makeIdentityPermutation(int64_t size) {
  std::vector<int64_t> perm(size);
  std::iota(perm.begin(), perm.end(), 0);
  return perm;
}

std::vector<int64_t> makePermutationFromFn(
    int64_t size,
    const fl::Dataset::PermutationFunction& fn) {
  if (!fn) {
    throw std::invalid_argument("PermutationFunction is null");
  }
  auto perm = makeIdentityPermutation(size);
  std::transform(perm.begin(), perm.end(), perm.begin(), fn);
  return perm;
}

} // namespace

namespace fl {

ResampleDataset::ResampleDataset(std::shared_ptr<const Dataset> dataset)
    : ResampleDataset(dataset, makeIdentityPermutation(dataset->size())) {}

ResampleDataset::ResampleDataset(
    std::shared_ptr<const Dataset> dataset,
    std::vector<int64_t> resamplevec)
    : dataset_(dataset) {
  if (!dataset_) {
    throw std::invalid_argument("dataset to be resampled is null");
  }
  resample(std::move(resamplevec));
}

ResampleDataset::ResampleDataset(
    std::shared_ptr<const Dataset> dataset,
    const PermutationFunction& fn,
    int n)
    : ResampleDataset(
          dataset,
          makePermutationFromFn(n == -1 ? dataset->size() : n, fn)) {}

void ResampleDataset::resample(std::vector<int64_t> resamplevec) {
  resampleVec_ = std::move(resamplevec);
}

std::vector<af::array> ResampleDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  return dataset_->get(resampleVec_[idx]);
}

int64_t ResampleDataset::size() const {
  return resampleVec_.size();
}

} // namespace fl
