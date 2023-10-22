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
 * Applies linear transformation on input: \f$y = Wx + b \f$.
 * This layer takes in an input of shape [`input_size`, *, *, *] and transforms
 * it to an output of shape [`output_size`, *, *, *].
 */
class FL_API Linear : public UnaryModule {
 private:
  Linear() = default; // Intentionally private

  int nIn_, nOut_;
  bool bias_;

  FL_SAVE_LOAD_WITH_BASE(UnaryModule, nIn_, nOut_, bias_)

  void initialize();

 public:
  /**
   * Constructs a Linear module from the input and output sample sizes.
   *
   * @param input_size the size of each input sample
   * @param output_size the size of each output sample
   * @param bias a boolean value that controls whether the layer will include
   *  a bias term \f$b\f$.
   */
  Linear(int input_size, int output_size, bool bias = true);

  /**
   * Constructs a Linear module from the weight parameter \f$w\f$. The layer
   * will not include the bias term \f$b\f$ in this case.
   *
   * @param w the 2D `Variable` tensor for the weight \f$w\f$.
   *  The shape should be [`output_size`, `input_size`].
   */
  explicit Linear(const Variable& w);

  /**
   * Constructs a Linear module from the weight parameter \f$w\f$ and the bias
   * parameter \f$b\f$.
   *
   * @param w the 2D `Variable` tensor for the weight \f$w\f$.
   *  The shape should be [`output_size`, `input_size`].
   * @param b the 1D `Variable` tensor for the bias \f$b\f$.
   *  The shape should be [`output_size`].
   */
  Linear(const Variable& w, const Variable& b);

  /**
   * Constructs an Linear module from another, performing a deep copy of the
   * parameters.
   *
   * @param other The Linear module to copy from.
   */
  Linear(const Linear& other);

  Linear& operator=(const Linear& other);

  Linear(Linear&& other) = default;

  Linear& operator=(Linear&& other) = default;

  Variable forward(const Variable& input) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Linear)
