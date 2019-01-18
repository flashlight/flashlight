/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/nn/modules/Module.h"

namespace fl {

/**
 * Applies Layer Normalization on a given input as described in the paper
 * [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf).
 *
 * The operation implemented is:
 * \f[out(x) = \frac{x - E[x]}{\sqrt{Var[x]+\epsilon}} \times \gamma + \beta \f]
 * where \f$E[x]\f$ and \f$Var[x]\f$ are the mean and variance of the input
 * \f$x\f$ calculated per specified axes, \f$\epsilon\f$ is a small value
 * added to the variance to avoid divide-by-zero, and \f$\gamma\f$ and
 * \f$\beta\f$ are learnable parameters for affine transformation.
 */
class LayerNorm : public UnaryModule {
 private:
  LayerNorm() = default;

  std::vector<int> featAxes_;
  double epsilon_;
  bool affine_;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule, featAxes_, epsilon_, affine_)

  void initialize();

 public:
  /**
   * Constructs a LayerNorm module.
   *
   * @param feat_axis the axis over which per-dimension normalization
   *  is performed.
   * @param eps \f$\epsilon\f$
   * @param affine a boolean value that controls the learning of \f$\gamma\f$
   *  and \f$\beta\f$. \f$\gamma\f$ and \f$\beta\f$ are set to 1, 0 respectively
   *  if set to `false`, or initialized as learnable parameters
   *  if set to `true`.
   */
  explicit LayerNorm(int feat_axis, double eps = 1e-5, bool affine = true);

  /**
   * Constructs a LayerNorm module.
   *
   * @param feat_axes the axes over which per-dimension normalization
   *  is performed. Usually set as the batch axis.
   * @param eps \f$\epsilon\f$
   * @param affine a boolean value that controls the learning of \f$\gamma\f$
   *  and \f$\beta\f$. \f$\gamma\f$ and \f$\beta\f$ are set to 1, 0 respectively
   *  if set to `false`, or initialized as learnable parameters
   *  if set to `true`.
   */
  explicit LayerNorm(
      const std::vector<int>& feat_axes,
      double eps = 1e-5,
      bool affine = true);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::LayerNorm)
