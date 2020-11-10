/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/AdaptiveEmbedding.h"

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

AdaptiveEmbedding::AdaptiveEmbedding(
    int embeddingDim,
    std::vector<int> cutoff,
    float divValue /*= 4 */)
    : embeddingDim_(embeddingDim), cutoff_(cutoff), divValue_(divValue) {
  if (cutoff_.empty()) {
    throw std::invalid_argument("Invalid cutoff for AdaptiveEmbedding");
  }
  double stdv = std::sqrt(1.0 / (double)embeddingDim_);
  // to be in agreement with the adaptive softmax to simplify
  // tied version of adaptive input and softmax
  auto headEmbedding = fl::normal(cutoff_[0], embeddingDim_, stdv, 0);
  params_.push_back(headEmbedding);
  auto head = fl::glorotUniform(
      af::dim4(embeddingDim_, embeddingDim_), embeddingDim_, embeddingDim_);
  params_.push_back(head);

  int denominator = 1;
  for (int tailIdx = 1; tailIdx < cutoff_.size(); tailIdx++) {
    denominator *= divValue_;
    int tailEmbeddingDim = embeddingDim_ / denominator;
    double stdvTail = std::sqrt(1.0 / (double)tailEmbeddingDim);
    // to be in agreement with the adaptive softmax to simplify
    // tied version of adaptive input and softmax
    auto tailEmbedding = fl::normal(
        cutoff_[tailIdx] - cutoff_[tailIdx - 1], tailEmbeddingDim, stdvTail, 0);
    params_.push_back(tailEmbedding);
    auto tail = fl::glorotUniform(
        af::dim4(embeddingDim_, tailEmbeddingDim),
        tailEmbeddingDim,
        embeddingDim_);
    params_.push_back(tail);
  }
}

Variable AdaptiveEmbedding::forward(const Variable& input) {
  auto flatInput = flat(input);
  std::vector<Variable> indices;
  std::vector<Variable> embeddings;

  af::array headMask = flatInput.array() < cutoff_[0];
  if (af::sum(headMask).scalar<unsigned int>() > 0) {
    auto headEmbedding =
        embedding(flatInput(headMask), reorder(params_[0], 1, 0));
    headEmbedding = matmul(params_[1], headEmbedding);
    indices.push_back(Variable(af::where(headMask), false));
    embeddings.push_back(headEmbedding);
  }

  for (int tailIdx = 1; tailIdx < cutoff_.size(); tailIdx++) {
    af::array tailMask = flatInput.array() < cutoff_[tailIdx] &&
        flatInput.array() >= cutoff_[tailIdx - 1];
    if (af::anyTrue<bool>(tailMask)) {
      auto tailEmbedding = embedding(
          flatInput(tailMask) - cutoff_[tailIdx - 1],
          reorder(params_[tailIdx * 2], 1, 0));
      tailEmbedding = matmul(params_[tailIdx * 2 + 1], tailEmbedding);
      indices.push_back(Variable(af::where(tailMask), false));
      embeddings.push_back(tailEmbedding);
    }
  }
  if (embeddings.size() == 0) {
    throw std::invalid_argument(
        "Invalid input, no positions in the AdaptiveEmbedding layer");
  }
  auto result = fl::concatenate(embeddings, 1);
  auto resultIndices = fl::concatenate(indices, 0);
  af::array tmpValues, tmpIndices;
  af::sort(tmpValues, tmpIndices, resultIndices.array(), 0);
  return moddims(
      result(af::span, tmpIndices),
      af::dim4(embeddingDim_, input.dims(0), input.dims(1), input.dims(2)));
}

std::string AdaptiveEmbedding::prettyString() const {
  std::ostringstream ss;
  ss << "AdaptiveEmbedding (dim: " << embeddingDim_ << "), (cutoff: ";
  for (int i = 0; i < cutoff_.size() - 1; i++) {
    ss << cutoff_[i] << ", ";
  }
  ss << cutoff_[cutoff_.size() - 1] << "), "
     << "(divValue: " << divValue_ << ")";
  return ss.str();
}

} // namespace fl
