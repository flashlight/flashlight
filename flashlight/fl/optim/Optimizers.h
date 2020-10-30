/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "flashlight/fl/autograd/Variable.h"

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
 private:
  /**
   * Serialize the module's parameters.
   */
  FL_SAVE_LOAD(lr_, parameters_)

 protected:
  std::vector<Variable> parameters_;
  double lr_;

  FirstOrderOptimizer() = default;

 public:
  /** The `FirstOrderOptimizer` base class constructor.
   * @param parameters The parameters from e.g. `model.parameters()`
   * @param learningRate The learning rate.
   */
  FirstOrderOptimizer(
      const std::vector<Variable>& parameters,
      double learningRate);

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
  virtual void zeroGrad();

  /**
   * Generates a stringified representation of the optimizer.
   *
   * @return a string containing the optimizer label
   */
  virtual std::string prettyString() const = 0;

  virtual ~FirstOrderOptimizer() = default;
};

} // namespace fl
