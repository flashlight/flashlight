#pragma once

#include "flashlight/fl/autograd/Variable.h"

namespace fl {
namespace ext {

enum ScaleFactorIncreaseForm { ADDITIVE = 0, MULTIPLICATIVE };

/**
 * Checks the input array, `in`, to determin if any of its elements are `nan` or
 * `inf`. If yes, it sets the `isInvalidArray` to one. For efficiency, when
 * compiled with the CUDA backend, it keeps the `isInvalidArray` on the GPU
 * side.
 *
 * @param in The input array that will be inspected.
 * @param isInvalidArray `isInvalidArray` will be set if contents of `in`
 * are invalid.
 */
void validityCheck(af::array& in, af::array& isInvalidArray);

/**
 * Scales the gard using the given `scaleFactor` if the value of
 * `isInvalidArray` is zero. Otherwise, resets the gradient values.
 *
 * @param grads The input gradients that will be scaled.
 * @param scaleFactor The coefficient that is used for scaling the gradients.
 * @param isInvalidArray If one, the gradients will not be scaled. They will be
 * set to zero.
 */
void scaleGrads(
    af::array& grads,
    af::array& scaleFactor,
    af::array& isInvalidArray);

/**
 * Scales the loss using the given `scaleFactor` as detailed in what follows
 * `loss` = `loss` * `scaleFactor`.
 *
 * @param loss The input loss that will be scaled.
 * @param scaleFactor The coefficient that is used for scaling the gradients.
 */
fl::Variable scaleLoss(fl::Variable& loss, fl::Variable& scaleFactor);

/**
 * Adjusts the `scaleFactor` by dividing it by two in case the value of
 * `isInvalidArray` is one. It also resets the value of `isInvalidArray` array.
 *
 * @param scaleFactor The coefficient that is used for scaling the gradients.
 * @param isInvalidArray If set two one, `scaleFactor` will be divided by 2.
 * @param minScaleFactor If `scaleFactor` / 2 < `minScaleFactor`, it will not be
 * scaled.
 */
bool decreaseScaleFactor(
    af::array& scaleFactor,
    af::array& isInvalidArray,
    const af::array& minScaleFactor);

/**
 * Adjusts the `scaleFactor` by multiplying or incrementing it by 2.
 *
 * @param scaleFactor The coefficient that is used for scaling the gradients.
 * @param maxScaleFactor The increase only occurs if the new value if smaller
 * than or equal to `maxScaleFactor`
 * @param increaseForm Depending on this value, increase can happen in two
 * fashions: additive or multiplicative.
 */
void increaseScaleFactor(
    af::array& scaleFactor,
    const af::array& maxScaleFactor,
    const ScaleFactorIncreaseForm& increaseForm);

} // namespace ext
} // namespace fl
