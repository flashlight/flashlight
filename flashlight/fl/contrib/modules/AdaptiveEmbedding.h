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
 * output from adaptive lookup from https://arxiv.org/pdf/1809.10853.pdf with
 * final shape
 * [`embedding_dim`, \f$B_1\f$, \f$B_2\f$ (optional), \f$B_3\f$ (optional)].
 */
class FL_API AdaptiveEmbedding : public UnaryModule {
 private:
  AdaptiveEmbedding() = default; // Intentionally private
  int embeddingDim_;
  std::vector<int> cutoff_;
  float divValue_;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule, embeddingDim_, cutoff_, divValue_)

 public:
  /**
   * Constructs an Embedding module.
   *
   * @param[in] embedding_dim the size of each embedding vector
   * @param[in] cutoff a sequence of integers sorted in ascending order, which
   * determines the relative size of each bucket, and how many partitions are
   * created. For example, given cutoffs `{5, 50, 100}`, the head bucket will
   * contain `5` targets, the
   * first tail bucket will contain `50 - 5 = 45` targets (subtracting the size
   * of the head bucket), the second tail bucket will contain `100 - 50 = 50`
   * targets (subtracting the size of the first tail bucket). Cutoffs must be
   * specified to accommodate all targets: any remaining targets are not
   * assigned to an 'overflow' bucket.
   * @param[in] divValue is the scaling factor for tail groups dimention
   * reduction (see paper https://arxiv.org/pdf/1809.10853.pdf for details).
   */
  explicit AdaptiveEmbedding(
      int embeddingDim,
      std::vector<int> cutoff,
      float divValue = 4);

  Variable forward(const Variable& input) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::AdaptiveEmbedding)
