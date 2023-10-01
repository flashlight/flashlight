/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/nn/modules/AdaptiveSoftMax.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

class Shape;
class Tensor;

/**
 * An efficient approximation of the softmax function and negative
 * log-likelihood loss. Computes the Adaptive Softmax, as given by [Grave et al
 * (2017)](https://arxiv.org/pdf/1609.04309.pdf): _Efficient softmax
 * approximation for GPUs_. Efficient when the number of classes over which the
 * softmax is being computed is very high and the label distribution is highly
 * imbalanced.
 *
 * Adaptive softmax buckets the inputs based on their frequency, where clusters
 * may be different number of targets each. For each minibatch, only clusters
 * for which at least one target is present are evaluated. Forward pass for
 * low-frequency inputs are approximated with lower rank matrices so as to speed
 * up computation.
 */
class AdaptiveSoftMaxLoss : public BinaryModule {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      BinaryModule,
      activation_,
      reduction_,
      fl::versioned(ignoreIndex_, 1))
  std::shared_ptr<AdaptiveSoftMax> activation_;
  ReduceMode reduction_;
  int ignoreIndex_{-1};

  Variable
  cast(const Variable& input, const Shape& outDims, const Tensor& indices);

 public:
  AdaptiveSoftMaxLoss() = default;

  /**
   * Create an `AdaptiveSoftMaxLoss` with given parameters
   *
   * @param reduction the reduction mode - see `ReductionMode` See
   * documentation on `ReduceMode` for available options.
   * @param ignoreIndex a target value that is ignored and does not contribute
   * to the loss or the input gradient. If `reduce` is MEAN, the loss is
   * averaged over non-ignored targets.
   */
  explicit AdaptiveSoftMaxLoss(
      std::shared_ptr<AdaptiveSoftMax> activation,
      ReduceMode reduction = ReduceMode::MEAN,
      int ignoreIndex = -1);
  std::shared_ptr<AdaptiveSoftMax> getActivation() const;

  /**
   * Computes the categorical cross entropy loss for some input and target
   * tensors (uses adaptive softmax function to do this efficiently)
   *
   * @param inputs a `Variable` with shape [\f$C\f$, \f$B_1\f$, \f$B_2\f$,
   * \f$B_3\f$] where \f$C\f$ is the number of classes.
   * @param targets an integer `Variable` with shape [\f$B_1\f$, \f$B_2\f$,
   * \f$B_3\f$]. The values must be in [\f$0\f$, \f$C - 1\f$]
   */
  Variable forward(const Variable& inputs, const Variable& targets) override;

  void setParams(const Variable& var, int position) override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::AdaptiveSoftMaxLoss)
CEREAL_CLASS_VERSION(fl::AdaptiveSoftMaxLoss, 1)
