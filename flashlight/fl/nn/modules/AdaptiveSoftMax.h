/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/nn/modules/Container.h"

namespace fl {

/**
 * An efficient approximation of the softmax function and negative
 * log-likelihood loss (see AdaptiveSoftmaxLoss). Computes the Adaptive Softmax,
 * as given by [Grave et al (2017)](https://arxiv.org/pdf/1609.04309.pdf):
 * _Efficient softmax approximation for GPUs_. Efficient when the number of
 * classes over which the softmax is being computed is very high and the label
 * distribution is highly imbalanced.
 *
 * Adaptive softmax buckets the inputs based on their frequency, where clusters
 * may be different number of targets each. For each minibatch, only clusters
 * for which at least one target is present are evaluated. Forward pass for
 * low-frequency inputs are approximated with lower rank matrices so as to speed
 * up computation.
 */
class FL_API AdaptiveSoftMax : public UnaryModule {
 private:
  FL_SAVE_LOAD_WITH_BASE(UnaryModule, cutoff_, divValue_)
  std::vector<int> cutoff_;
  float divValue_;

  /**
   * Compute the output of the entire distribution.
   *
   * @param inputs values for each class to compute probabilities over
   * @param head_output the output of the first frequency bucket (the 'top'
   * bucket)
   * @returns `Variable` containing the log probabilities over the full
   * distribution
   */
  Variable getFullLogProb(const Variable& inputs, const Variable& headOutput)
      const;

 public:
  AdaptiveSoftMax() = default;

  /**
   * Create an `AdaptiveSoftMax` with given parameters
   *
   * @param input_size the size of the input tensor, which doesn't has to be the
   * number of classes.
   * @param cutoff a sequence of integers sorted in ascending order, which
   * determines the relative size of each bucket, and how many partitions are
   * created. For example, given cutoffs `{5, 50, 100}`, the head bucket will
   * contain `5 + 2 = 7` targets (`2` additional from the two tail buckets), the
   * first tail bucket will contain `50 - 5 = 45` targets (subtracting the size
   * of the head bucket), the second tail bucket will contain `100 - 50 = 50`
   * targets (subtracting the size of the first tail bucket). Cutoffs must be
   * specified to accommodate all targets: any remaining targets are not
   * assigned to an 'overflow' bucket.
   * @param div_value determines the number of hidden units in the intermediate
   * layer for each tail bucket:
   * \f[
   *    \left\lfloor \frac{input\_size}{div\_value^{idx}} \right\rfloor
   * \f]
   */
  AdaptiveSoftMax(
      int inputSize,
      const std::vector<int>& cutoff,
      float divValue = 4);

  /**
   * Computes log-probabilities across all classes for some input.
   *
   * @param inputs a Variable with size [\f$C_{in}\f$, \f$B_1\f$, \f$B_2\f$,
   * \f$B_3\f$]
   * @return a Variable containing log probabilities for each class with size
   * [\f$C\f$, \f$B_1\f$, \f$B_2\f$, \f$B_3\f$], where \f$C\f$ is the number of
   * classes.
   */
  Variable forward(const Variable& inputs) override;

  /**
   * Computes the class with highest probability for each example in a given
   * input.
   *
   * @param inputs a Variable with size [\f$C_{in}\f$, \f$B_1\f$, \f$B_2\f$,
   * \f$B_3\f$].
   * @return a Variable with shape [\f$1\f$, \f$B_1\f$, \f$B_2\f$, \f$B_3\f$],
   * containing the classes with the highest probabilities, over each sample.
   */
  Variable predict(const Variable& inputs) const;
  std::vector<int> getCutoff() const;

  std::unique_ptr<Module> clone() const override;
  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::AdaptiveSoftMax)
