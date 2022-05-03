/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/nn/modules/Embedding.h"

#include <cmath>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"

namespace fl {

Embedding::Embedding(int embeddingDim, int numEmbeddings)
    : embeddingDim_(embeddingDim), numEmbeddings_(numEmbeddings) {
  initialize();
}

Embedding::Embedding(const Variable& w)
    : UnaryModule({w}), embeddingDim_(w.dim(0)), numEmbeddings_(w.dim(1)) {}

void Embedding::initialize() {
  double stdv = std::sqrt(1.0 / (double)embeddingDim_);
  auto embeddings =
      uniform(embeddingDim_, numEmbeddings_, -stdv, stdv, fl::dtype::f32, true);
  params_ = {embeddings};
}

Variable Embedding::forward(const Variable& input) {
  return embedding(input, params_[0]);
}

std::string Embedding::prettyString() const {
  std::ostringstream ss;
  ss << "Embedding (embeddings: " << numEmbeddings_
     << ") (dim: " << embeddingDim_ << ")";
  return ss.str();
}

} // namespace fl
