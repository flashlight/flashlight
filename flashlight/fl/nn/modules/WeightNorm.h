/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/modules/Container.h"
#include "flashlight/fl/nn/modules/Conv2D.h"
#include "flashlight/fl/nn/modules/Linear.h"

namespace fl {

/** A weight normalization layer. This layer wraps a given module to create a
 * weight normalized implementation of the module. WeightNorm currently
 * supports Linear and Conv2D. For example:
 *
 * \code
 * WeightNorm wn(Linear(128, 128), 0);
 * \endcode
 *
 * For more details see [Weight Normalization: A Simple Reparameterization to
 * Accelerate Training of Deep Neural Networks](
 * https://arxiv.org/abs/1602.07868)
 */
class FL_API WeightNorm : public Module {
 private:
  WeightNorm() = default;

  std::shared_ptr<Module> module_;

  // Computes the norm over all dimensions except dim_
  int dim_;
  std::vector<int> normDim_;

  void transformDims();

  void computeWeight();

  void initParams();

  FL_SAVE_LOAD_DECLARE()

 public:
  /** Construct a WeightNorm layer.
   * @param module A module to wrap (must be one of Linear or Conv2D). Takes
   * ownership of the module.
   * @param dim The dimension to normalize.
   */
  template <class T>
  WeightNorm(T&& module, int dim)
      : WeightNorm(
            std::make_shared<std::decay_t<T>>(std::forward<T>(module)),
            dim) {}

  /** Construct a WeightNorm layer.
   * @param module Shared pointer to a module to wrap (the module must be one
   * of Linear or Conv2D)
   * @param dim The dimension to normalize.
   */
  template <class T>
  WeightNorm(std::shared_ptr<T> module, int dim) : module_(module), dim_(dim) {
    transformDims();
    initParams();
  }

  /**
   * Construct a WeightNorm module from another, performing a deep copy for the
   * wrapped module.
   *
   * @param other The WeightNorm module to copy from.
   */
  WeightNorm(const WeightNorm& other);

  /**
   * Construct a WeightNorm module from another, performing a deep copy for the
   * wrapped module.
   *
   * @param other The WeightNorm module to copy from.
   */
  WeightNorm& operator=(const WeightNorm& other);

  WeightNorm(WeightNorm&& other) = default;

  WeightNorm& operator=(WeightNorm&& other) = default;

  /**
   * Returns a pointer to the inner `Module` normalized by this `WeightNorm`.
   *
   * @return a module pointer.
   */
  ModulePtr module() const;

  void train() override;

  void eval() override;

  void setParams(const Variable& var, int position) override;

  std::vector<Variable> forward(const std::vector<Variable>& inputs) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

template <class Archive>
void WeightNorm::save(Archive& ar, const uint32_t /* version */) const {
  // Not saving weight since it can be inferred from from v and g.
  auto wt = module_->param(0);
  module_->setParams(Variable(), 0);
  ar(cereal::base_class<Module>(this), module_, dim_, normDim_);
  module_->setParams(wt, 0);
}

template <class Archive>
void WeightNorm::load(Archive& ar, const uint32_t /* version */) {
  ar(cereal::base_class<Module>(this), module_, dim_, normDim_);
  computeWeight();
}

} // namespace fl

CEREAL_REGISTER_TYPE(fl::WeightNorm)
CEREAL_REGISTER_POLYMORPHIC_RELATION(fl::Module, fl::WeightNorm)
