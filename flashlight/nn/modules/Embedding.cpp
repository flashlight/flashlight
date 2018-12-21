/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Embedding.h"

#include <flashlight/autograd/Functions.h>
#include <flashlight/nn/Init.h>

namespace fl {

Embedding::Embedding(int embedding_dim, int num_embeddings)
    : embeddingDim_(embedding_dim), numEmbeddings_(num_embeddings) {
  initialize();
}

void Embedding::initialize() {
  double stdv = std::sqrt(1.0 / (double)embeddingDim_);
  auto embeddings = uniform(embeddingDim_, numEmbeddings_, -stdv, stdv);
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
