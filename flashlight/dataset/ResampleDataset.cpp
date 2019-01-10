/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/dataset/ResampleDataset.h"

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
    const PermutationFunction& fn)
    : ResampleDataset(dataset, makePermutationFromFn(dataset->size(), fn)) {}

void ResampleDataset::resample(std::vector<int64_t> resamplevec) {
  if (size() != resamplevec.size()) {
    throw std::invalid_argument("wrong vector size for `resample`");
  }
  resampleVec_ = std::move(resamplevec);
}

std::vector<af::array> ResampleDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  return dataset_->get(resampleVec_[idx]);
}

int64_t ResampleDataset::size() const {
  return dataset_->size();
}

} // namespace fl
