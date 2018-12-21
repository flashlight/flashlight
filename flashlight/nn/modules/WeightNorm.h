/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/autograd/Functions.h>
#include <flashlight/common/Exception.h>
#include <flashlight/nn/Init.h>
#include "Container.h"

#include "Conv2D.h"
#include "Linear.h"

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
class WeightNorm : public Container {
 private:
  WeightNorm() = default;

  // Computes the norm over all dimensions except dim_
  int dim_;
  std::vector<int> normDim_;

  void transformDims();

  void computeWeight();

  FL_SAVE_LOAD_DECLARE()

 public:
  /** Construct a WeightNorm layer.
   * @param module A module to wrap (must be one of Linear or Conv2D)
   * @param dim The dimension to normalize.
   */
  template <class T>
  WeightNorm(const T& module, int dim)
      : WeightNorm(std::make_shared<T>(module), dim) {}

  /** Construct a WeightNorm layer.
   * @param module Shared pointer to a module to wrap (the module must be one
   * of Linear or Conv2D)
   * @param dim The dimension to normalize.
   */
  template <class T>
  WeightNorm(std::shared_ptr<T> module, int dim) : dim_(dim) {
    add(module);

    auto parent_params = modules_[0]->params();
    auto v = parent_params[0];
    transformDims();
    auto g = Variable(norm(v, normDim_).array(), true);
    if (parent_params.size() == 2) {
      auto b = parent_params[1];
      params_ = {v, g, b};
    } else if (parent_params.size() == 1) {
      params_ = {v, g};
    } else {
      AFML_THROW_ERR(
          "[WeightNorm] Only support Linear and Conv2D.\n", AF_ERR_ARG);
    }
  }

  void eval() override;

  void setParams(const Variable& var, int position) override;

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};

template <class Archive>
void WeightNorm::save(Archive& ar, const uint32_t /* version */) const {
  // Not saving weight since it can inferred from from v and g.
  auto wt = modules_[0]->param(0);
  modules_[0]->setParams(Variable(), 0);
  ar(cereal::base_class<Container>(this), dim_, normDim_);
  modules_[0]->setParams(wt, 0);
}

template <class Archive>
void WeightNorm::load(Archive& ar, const uint32_t /* version */) {
  ar(cereal::base_class<Container>(this), dim_, normDim_);
  computeWeight();
}

} // namespace fl

CEREAL_REGISTER_TYPE(fl::WeightNorm)
CEREAL_REGISTER_POLYMORPHIC_RELATION(fl::Container, fl::WeightNorm)
