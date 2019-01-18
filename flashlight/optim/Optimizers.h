/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/


#pragma once

#include <vector>

#include <arrayfire.h>

#include "flashlight/autograd/Variable.h"

namespace fl {

/** An abstract base class for first-order gradient-based optimizers. Any
 * derived class must implement the step() function.
 * Example usage:
 *
 * \code
 * SGDOptimizer optimizer(model.parameters(), 1e-1);
 * auto loss = model(data);
 * loss.backward();
 * optimizer.step();
 * optimizer.zeroGrad();
 * \endcode
 */
class FirstOrderOptimizer {
 protected:
  std::vector<Variable> parameters_;
  double lr_;

 public:
  /** The `FirstOrderOptimizer` base class constructor.
   * @param parameters The parameters from e.g. `model.parameters()`
   * @param learning_rate The learning rate.
   */
  FirstOrderOptimizer(
      const std::vector<Variable>& parameters, double learning_rate);

  virtual void step() = 0;

  /** Get the learning rate. */
  double getLr() const {
    return lr_;
  }

  /** Set the learning rate. */
  void setLr(double lr) {
    lr_ = lr;
  }

  /** Zero the gradients for all the parameters being optimized. Typically
   * this will be called after every call to step().
   */
  void zeroGrad();

  virtual ~FirstOrderOptimizer() = default;
};

/** A Stochastic Gradient Descent (SGD) optimizer. At its most basic this
 * implements the update
 * \f[
 *   w = w - lr * g
 * \f]
 *
 * When momentum is used the update becomes
 * \f[
 *   v &= \rho * v + g \\
 *   w &= w - lr * v
 * \f]
 */
class SGDOptimizer : public FirstOrderOptimizer {
  bool useNesterov_;
  double mu_;
  double wd_;
  std::vector<af::array> velocities_;

 public:
  /** SGDOptimizer constructor.
   * @param parameters The parameters from e.g. `model.parameters()`
   * @param learning_rate The learning rate.
   * @param momentum The momentum.
   * @param weight_decay The amount of L2 weight decay to use for all the
   * parameters.
   * @param use_nesterov Whether or not to use nesterov style momentum.
   */
  SGDOptimizer(
      const std::vector<Variable>& parameters,
      double learning_rate,
      double momentum = 0,
      double weight_decay = 0,
      bool use_nesterov = false);
  void step() override;
};

/** An implementation of the Adam optimizer.
 * For more details see the paper
 * [Adam: A Method for Stochastic Optimization](
 *    https://arxiv.org/abs/1412.6980).
 */
class AdamOptimizer : public FirstOrderOptimizer {
  double beta1_;
  double beta2_;
  double eps_;
  double wd_;
  int count_;
  std::vector<af::array> biasedFirst_;
  std::vector<af::array> biasedSecond_;

 public:
  /** Construct an Adam optimizer.
   * @param parameters The parameters from e.g. `model.parameters()`.
   * @param learning_rate The learning rate.
   * @param beta1 Adam hyperparameter \f$ \beta_1 \f$.
   * @param beta2 Adam hyperparameter \f$ \beta_2 \f$.
   * @param epsilon A small value used for numerical stability.
   * @param weight_decay The amount of L2 weight decay to use for all the
   * parameters.
   */
  AdamOptimizer(
      const std::vector<Variable>& parameters,
      double learning_rate,
      double beta1 = 0.9,
      double beta2 = 0.999,
      double epsilon = 1e-8,
      double weight_decay = 0);
  void step() override;
};

/** An implementation of the RMSProp optimizer. For more details see Geoff
 * Hinton's [lecture slides](
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf).
 */
class RMSPropOptimizer : public FirstOrderOptimizer {
  bool useFirst_;
  double rho_;
  double eps_;
  double wd_;
  std::vector<af::array> first_;
  std::vector<af::array> second_;

 public:
  /** Construct an RMSProp optimizer.
   * @param parameters The parameters from e.g. `model.parameters()`.
   * @param learning_rate The learning rate.
   * @param rho The weight in the term \f$ rho * m + (1-rho) * g^2 \f$.
   * @param epsilon A small value used for numerical stability.
   * @param weight_decay The amount of L2 weight decay to use for all the
   * parameters.
   * @param use_first Use the first moment in the update. When `true` keep
   * a running mean of the gradient and subtract it from the running mean of
   * the squared gradients.
   */
  RMSPropOptimizer(
      const std::vector<Variable>& parameters,
      double learning_rate,
      double rho = 0.99,
      double epsilon = 1e-8,
      double weight_decay = 0,
      bool use_first = false);
  void step() override;
};

} // namespace fl
