/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Container.h"
#include "flashlight/fl/nn/modules/Conv2D.h"
#include "flashlight/fl/nn/modules/LayerNorm.h"
#include "flashlight/fl/nn/modules/Linear.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * A module which implements a Conformer block (we use LayerNorm everywhere).
 *
 * For details, see [Gulati et al
 * (2020)](https://arxiv.org/pdf/2005.08100.pdf).
 *
 * Input dimension at forward is assumed to be CxTxBx1, where C is the
 * number of features, T the sequence length and B the batch size.
 * @param modelDim input embedding dimension
 * @param headDim dimension of each head
 * @param mlpDim dimension of the feed-forward layers
 * @param nHeads number of heads
 * @param posEmbContextSize size for learnt relative positional
 * embedding matrix (2 * posEmbContextSize - 1) * nHeads
 * @param convKernelSize convolution layers kernel
 * @param pDropout dropout probability
 * @param pLayerdrop layer dropout probability
 */
class Conformer : public Container {
 public:
  explicit Conformer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      int32_t posEmbContextSize,
      int32_t convKernelSize,
      float pDropout,
      float pLayerDropout = 0.);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;
  std::string prettyString() const override;

 private:
  int32_t nHeads_;
  int32_t posEmbContextSize_;
  int32_t convKernelSize_;
  double pDropout_;
  float pLayerDropout_;

  std::shared_ptr<Linear> w11_, w12_, w21_, w22_, wq_, wk_, wv_, wf_, conv1_, conv2_;
  std::shared_ptr<LayerNorm> norm1_, norm2_, normMhsa_, normConv1_, normConv2_,
      norm3_;
  std::shared_ptr<Conv2D> convDepthWise_;

  static Variable conformerInitLinear(int32_t inDim, int32_t outDim);
  Variable mhsa(const Variable& input, const Variable& inputPadMask);
  Variable conv(const Variable& input);

  Conformer() = default;

  FL_SAVE_LOAD_WITH_BASE(
      Container,
      w11_,
      w12_,
      w21_,
      w22_,
      wq_,
      wk_,
      wv_,
      wf_,
      normMhsa_,
      norm1_,
      norm2_,
      norm3_,
      normConv1_,
      normConv2_,
      conv1_,
      conv2_,
      convDepthWise_,
      nHeads_,
      pDropout_,
      pLayerDropout_,
      posEmbContextSize_,
      convKernelSize_)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Conformer);
