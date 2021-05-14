#pragma once

#include "flashlight/fl/autograd/Variable.h"

namespace fl {
namespace ext {

/**
 * Checks the input array, `in`, to determin if any of its elements are `nan` or
 * `inf`. If yes, it sets the flag to one. For efficiency, when compiled with
 * the CUDA backend, it keeps the flag on the GPU side.
 * 
 * @param in The input array that will be inspected.
 * @param flag The flag the will be set if contents of `in` are invalid.
 */
void validityCheck(af::array& in, af::array& flag);

/**
 * Scales the gard using the given `scaleFactor` if the value of flag is zero.
 * Otherwise, resets the gradient values.
 *
 * @param grads The input gradients that will be scaled.
 * @param scaleFactor The coefficient that is used for scaling the gradients.
 * @param flag If one, the gradients will not be scaled. They will be set
 * to zero.
 */
void scaleGrads(af::array& grads, af::array& scaleFactor, af::array& flag);

/**
 * Scales the loss using the given `scaleFactor` as detailed in what follows
 * `loss` = `loss` * `scaleFactor`.
 *
 * @param loss The input loss that will be scaled.
 * @param scaleFactor The coefficient that is used for scaling the gradients.
 */
fl::Variable scaleLoss(fl::Variable& loss, fl::Variable& scaleFactor);

/**
 * Adjusts the `scaleFactor` by dividing it by two in case the value of `flag`
 * is one. It also resets the flag.
 *
 * @param scaleFactor The coefficient that is used for scaling the gradients.
 * @param flag If set two one, `scaleFactor` will be divided by 2.
 */
bool adjustScaleFactor(af::array& scaleFactor, af::array& flag);

} // namespace ext
} // namespace fl
