/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * Looks up embeddings from a learnable dictionary of fixed size.
 * This layer expects as input a list of indices with at most three dimensions,
 * [\f$B_1\f$, \f$B_2\f$ (optional), \f$B_3\f$ (optional)], and generates an
 * output from lookup of shape
 * [`embeddingDim`, \f$B_1\f$, \f$B_2\f$ (optional), \f$B_3\f$ (optional)].
 */
class FL_API Embedding : public UnaryModule {
 private:
  Embedding() = default; // Intentionally private

  int embeddingDim_;
  int numEmbeddings_;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule, embeddingDim_, numEmbeddings_)

  void initialize();

 public:
  /**
   * Constructs an Embedding module.
   *
   * @param embeddingDim the size of each embedding vector
   * @param numEmbeddings the size of the dictionary of embeddings
   */
  Embedding(int embeddingDim, int numEmbeddings);

  /**
   * Constructs an Embedding module from the weight parameter \f$w\f$.
   *
   * @param w the 2D `Variable` tensor for the weight \f$w\f$.
   *  The shape should be [`embeddingDim`, `numEmbeddings`].
   */
  explicit Embedding(const Variable& w);

  /**
   * Constructs an Embedding module from another, performing a copy of the
   * parameters.
   *
   * @param other The Embedding module to copy from.
   */
  Embedding(const Embedding& other);

  /**
   * Constructs an Embedding module from another, performing a copy of the
   * parameters.
   *
   * @param other The Embedding module to copy from.
   */
  Embedding& operator=(const Embedding& other);

  Embedding(Embedding&& other) = default;

  Embedding& operator=(Embedding&& other) = default;

  Variable forward(const Variable& input) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Embedding)
