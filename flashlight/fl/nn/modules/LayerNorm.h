/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

constexpr const int kLnVariableAxisSize = -1;
/**
 * Applies Layer Normalization on a given input as described in the paper
 * [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf).
 *
 * The operation implemented is:
 * \f[out(x) = \frac{x - E[x]}{\sqrt{Var[x]+\epsilon}} \times \gamma + \beta \f]
 * where \f$E[x]\f$ and \f$Var[x]\f$ are the mean and variance of the input
 * \f$x\f$ calculated along specified axis, \f$\epsilon\f$ is a small value
 * added to the variance to avoid divide-by-zero, and \f$\gamma\f$ and
 * \f$\beta\f$ are learnable parameters for affine transformation.
 */
class LayerNorm : public UnaryModule {
 public:
  /**
   * Constructs a LayerNorm module.
   *
   * @param axis the axis along which normalization is computed. Usually set as
   * the feature axis.
   * @param eps \f$\epsilon\f$
   * @param affine a boolean value that controls the learning of \f$\gamma\f$
   *  and \f$\beta\f$. \f$\gamma\f$ and \f$\beta\f$ are set to 1, 0 respectively
   *  if set to `false`, or initialized as learnable parameters
   *  if set to `true`.
   * @param axisSize total size of features specified by `axis` to perform
   *  elementwise affine transform. If the feat size is variable, use
   *  `kLnVariableAxisSize` which uses singleton weight, bias and tiles them
   *  dynamically according to the given input.
   */
  explicit LayerNorm(
      int axis,
      double eps = 1e-5,
      bool affine = true,
      int axisSize = kLnVariableAxisSize);

  /**
   * Constructs a LayerNorm module.
   *
   * @param axis the axis along which normalization is computed. Usually set as
   * the feature axis.
   * @param eps \f$\epsilon\f$
   * @param affine a boolean value that controls the learning of \f$\gamma\f$
   *  and \f$\beta\f$. \f$\gamma\f$ and \f$\beta\f$ are set to 1, 0 respectively
   *  if set to `false`, or initialized as learnable parameters
   *  if set to `true`.
   * @param axisSize total size of features specified by `axis` to perform
   *  elementwise affine transform. If the feat size is variable, use
   *  `kLnVariableAxisSize` which uses singleton weight, bias and tiles them
   *  dynamically according to the given input.
   */
  explicit LayerNorm(
      const std::vector<int>& axis,
      double eps = 1e-5,
      bool affine = true,
      int axisSize = kLnVariableAxisSize);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;

 private:
  LayerNorm() = default;

  // For legacy reasons, we store the complement of `axis`
  // to not break serialization
  std::vector<int> axisComplement_;
  double epsilon_;
  bool affine_;
  int axisSize_{kLnVariableAxisSize};

  FL_SAVE_LOAD_WITH_BASE(
      UnaryModule,
      axisComplement_,
      epsilon_,
      affine_,
      fl::versioned(axisSize_, 1))

  void initialize();
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::LayerNorm)
CEREAL_CLASS_VERSION(fl::LayerNorm, 1)
