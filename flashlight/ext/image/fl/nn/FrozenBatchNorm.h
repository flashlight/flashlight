/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/BatchNorm.h"

namespace fl {

/**
 * Applies Batch Normalization on a given input as described in the paper
 * [Batch Normalization: Accelerating Deep Network Training by Reducing Internal
 * Covariate Shift](https://arxiv.org/abs/1502.01852).
 *
 * The operation implemented is:
 * \f[out(x) = \frac{x - E[x]}{\sqrt{Var[x]+\epsilon}} \times \gamma + \beta \f]
 * where \f$E[x]\f$ and \f$Var[x]\f$ are the mean and variance of the input
 * \f$x\f$ calculated over the specified axis, \f$\epsilon\f$ is a small value
 * added to the variance to avoid divide-by-zero, and \f$\gamma\f$ and
 * \f$\beta\f$ are learnable parameters for affine transformation.
 */
class FrozenBatchNorm : public BatchNorm {
 private:
  FrozenBatchNorm() = default; // intentionally private
  FL_SAVE_LOAD_WITH_BASE(
      BatchNorm)

 public:
  /**
   * Constructs a FrozenBatchNorm module.
   *
   * @param featAxis the axis over which normalizationis performed
   * @param featSize the size of the dimension along `featAxis`
   * @param momentum an exponential average factor used to compute running mean
   *  and variance.
   *  \f[ runningMean = runningMean \times (1-momentum)
   *  + newMean \times momentum \f]
   *  If < 0, cumulative moving average is used.
   * @param eps \f$\epsilon\f$
   * @param affine a boolean value that controls the learning of \f$\gamma\f$
   *  and \f$\beta\f$. \f$\gamma\f$ and \f$\beta\f$ are set to 1, 0 respectively
   *  if set to `false`, or initialized as learnable parameters
   *  if set to `true`.
   * @param trackStats a boolean value that controls whether to track the
   *  running mean and variance while in train mode. If `false`, batch
   *  statistics are used to perform normalization in both train and eval mode.
   */
  FrozenBatchNorm(
      int featAxis,
      int featSize,
      double momentum = 0.1,
      double eps = 1e-5,
      bool affine = true,
      bool trackStats = true);

  /**
   * Constructs a FrozenBatchNorm module.
   *
   * @param featAxis the axis over which  normalization is performed
   * @param featSize total dimension along `featAxis`.
   *  For example, to perform Temporal Batch Normalization on input of size
   *  [\f$L\f$, \f$C\f$, \f$N\f$], use `featAxis` = {1}, `featSize` = \f$C\f$.
   *  To perform normalization per activation on input of size
   *  [\f$W\f$, \f$H\f$, \f$C\f$, \f$N\f$], use `featAxis` = {0, 1, 2},
   *  `featSize` = \f$W \times H \times C\f$.
   * @param momentum an exponential average factor used to compute running mean
   *  and variance.
   *  \f[ runningMean = runningMean \times (1-momentum)
   *  + newMean \times momentum \f]
   *  If < 0, cumulative moving average is used.
   * @param eps \f$\epsilon\f$
   * @param affine a boolean value that controls the learning of \f$\gamma\f$
   *  and \f$\beta\f$. \f$\gamma\f$ and \f$\beta\f$ are set to 1, 0 respectively
   *  if set to `false`, or initialized as learnable parameters
   *  if set to `true`.
   * @param trackStats a boolean value that controls whether to track the
   *  running mean and variance while in train mode. If `false`, batch
   *  statistics are used to perform normalization in both train and eval mode.
   */
  FrozenBatchNorm(
      const std::vector<int>& featAxis,
      int featSize,
      double momentum = 0.1,
      double eps = 1e-5,
      bool affine = true,
      bool trackStats = true);

  Variable forward(const Variable& input) override;

  void setRunningVar(const fl::Variable& x);

  void setRunningMean(const fl::Variable& x);

  void train() override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::FrozenBatchNorm)
