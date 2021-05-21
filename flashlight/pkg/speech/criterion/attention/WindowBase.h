/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace app {
namespace asr {

/**
 * Pretraining window class which defines attention mask
 * while attention is not trained yet, so it force aligning attention
 * thus we can have good proxy for attention between encoder-decoder
 * right at the beginning
 */
class WindowBase {
 public:
  WindowBase() {}

  /**
   * Compute window for the data for particular output step using
   * @param prevAttn previous step attention
   * @param step decoder step
   * @param inputSteps encoder output / decoder input length (max in the batch)
   * @param batchSize batch size
   * @param inputSizes actual encoder output / decoder input sizes (even before
   * the encoder, we need only the proportions); can be empty which means all
   * are treated to be the same size, size is 1xB
   * @param targetSizes actual decoder target output sizes (excluding padding);
   * can be empty
   */
  virtual Variable computeWindow(
      const Variable& prevAttn,
      int step,
      int targetLen,
      int inputSteps,
      int batchSize,
      const af::array& inputSizes = af::array(),
      const af::array& targetSizes = af::array()) const = 0;

  /**
   * Compute window for the data for entire decoder known target size
   * @param targetLen target size (max in the batch)
   * @param inputSteps encoder output / decoder input length (max in the batch)
   * @param batchSize batch size
   * @param inputSizes actual encoder output / decoder input sizes (even before
   * the encoder, we need only the proportions); can be empty which means all
   * are treated to be the same size, size is 1xB
   * @param targetSizes actual decoder target output sizes (excluding padding);
   * can be empty
   */
  virtual Variable computeVectorizedWindow(
      int targetLen,
      int inputSteps,
      int batchSize,
      const af::array& inputSizes = af::array(),
      const af::array& targetSizes = af::array()) const = 0;

  virtual ~WindowBase() {}

 protected:
  /**
   * Compute necessary matrix to process the padding later from the input sizes
   * @param inputSizes actual encoder output / decoder input sizes (even before
   * the encoder, we need only the proportions); can be empty which means all
   * are treated to be the same size, size is 1xB
   * @param inputSteps encoder output / decoder input length (max in the batch)
   * @param batchSize batch size
   * @param decoderStepsDim max decoder steps
   * @param doTile Do necessary tile to (decoderStepsDim, inputSteps, BatchSize)
   * or return (1, 1, BatchSize) vector (depends on the window we need to use)
   */
  af::array computeInputNotPaddedSize(
      const af::array& inputSizes,
      int inputSteps,
      int batchSize,
      int decoderStepsDim,
      bool doTile) const;

  /**
   * Compute necessary matrix to process the padding later from the target sizes
   * @param targetSizes actual decoder target output sizes (excluding padding);
   * can be empty
   * @param inputSteps encoder output / decoder input length (max in the batch)
   * @param targetLen target size (max in the batch)
   * @param batchSize batch size
   * @param decoderStepsDim max decoder steps
   */
  af::array computeTargetNotPaddedSize(
      const af::array& targetSizes,
      int inputSteps,
      int targetLen,
      int batchSize,
      int decoderStepsDim) const;

 private:
  FL_SAVE_LOAD()
};
} // namespace asr
} // namespace app
} // namespace fl
