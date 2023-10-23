/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
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

Embedding::Embedding(const Embedding& other)
    : UnaryModule(other.copyParams()),
      embeddingDim_(other.embeddingDim_),
      numEmbeddings_(other.numEmbeddings_) {
  train_ = other.train_;
}

Embedding& Embedding::operator=(const Embedding& other) {
  params_ = other.copyParams();
  train_ = other.train_;
  embeddingDim_ = other.embeddingDim_;
  numEmbeddings_ = other.numEmbeddings_;
  return *this;
}

void Embedding::initialize() {
  double stdv = std::sqrt(1.0 / (double)embeddingDim_);
  auto embeddings =
      uniform(embeddingDim_, numEmbeddings_, -stdv, stdv, fl::dtype::f32, true);
  params_ = {embeddings};
}

Variable Embedding::forward(const Variable& input) {
  return embedding(input, params_[0]);
}

std::unique_ptr<Module> Embedding::clone() const {
  return std::make_unique<Embedding>(*this);
}

std::string Embedding::prettyString() const {
  std::ostringstream ss;
  ss << "Embedding (embeddings: " << numEmbeddings_
     << ") (dim: " << embeddingDim_ << ")";
  return ss.str();
}

} // namespace fl
