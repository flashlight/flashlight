/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>

#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {
namespace pkg {
namespace vision {

/*
 * Resizes the smallest length edge of an image to be resize while keeping
 * the aspect ratio
 */
Tensor resizeSmallest(const Tensor& in, const int resize);

/*
 * Resize both sides of image to be length
 * @param resize`
 * will change aspect ratio
 */
Tensor resize(const Tensor& in, const int resize);

/*
 * Crop image @param in, starting from position @param x and @param y
 * with a target width and height of @param w and @param h respectively
 */
Tensor
crop(const Tensor& in, const int x, const int y, const int w, const int h);

/*
 * Take a center crop of image @param in,
 * where both image sides with be of length @param size
 */
Tensor centerCrop(const Tensor& in, const int size);

/*
 * Rotate an image
 * @param theta to which degree (in radius) a image will rotate
 * @param fillImg filling values on the empty spots
 */
Tensor rotate(const Tensor& input, const float theta, const Tensor& fillImg);

/*
 * Skew an image on the first dimension
 * @param theta to which degree (in radius) a image will skew
 * @param fillImg filling values on the empty spots
 */
Tensor skewX(const Tensor& input, const float theta, const Tensor& fillImg);

/*
 * Skew an image on the second dimension
 * @param theta to which degree (in radius) a image will skew
 * @param fillImg filling values on the empty spots
 */
Tensor skewY(const Tensor& input, const float theta, const Tensor& fillImg);

/*
 * Translate an image on the first dimension
 * @param shift number of pixels a image will translate
 * @param fillImg filling values on the empty spots
 */
Tensor translateX(const Tensor& input, const int shift, const Tensor& fillImg);

/*
 * Translate an image on the second dimension
 * @param shift number of pixels a image will translate
 * @param fillImg filling values on the empty spots
 */
Tensor translateY(const Tensor& input, const int shift, const Tensor& fillImg);

/*
 * Enhance the color of an image
 * @param enhance to which extend the color will change.
 */
Tensor colorEnhance(const Tensor& input, const float enhance);

/*
 * Remaps the image so that the darkest pixel becomes black (0), and the
 * lightest becomes white (255).
 */
Tensor autoContrast(const Tensor& input);

/*
 * Enhance the contrast of an image
 * @param enhance to which extend the contrast will change.
 */
Tensor contrastEnhance(const Tensor& input, const float enhance);

/*
 * Enhance the brightness of an image
 * @param enhance to which extend the brightness will change.
 */
Tensor brightnessEnhance(const Tensor& input, const float enhance);

/*
 * Enhance the sharpness of an image
 * @param enhance to which extend the sharpness will change.
 */
Tensor sharpnessEnhance(const Tensor& input, const float enhance);

/*
 * Invert each pixel of the image
 */
Tensor invert(const Tensor& input);

/*
 * Invert all pixel values above a threshold.
 */
Tensor solarize(const Tensor& input, const float threshold);

/*
 * Increase all pixel values below a threshold.
 */
Tensor
solarizeAdd(const Tensor& input, const float threshold, const float addValue);

/*
 * Applies a non-linear mapping to the input image, in order to create a uniform
 * distribution of grayscale values in the output image.
 */
Tensor equalize(const Tensor& input);

/*
 * Reduce the number of bits for each color channel.
 */
Tensor posterize(const Tensor& input, const int bitsToKeep);

/*
 * Transform a target array with label indices into a one-hot matrix
 */
Tensor
oneHot(const Tensor& targets, const int numClasses, const float labelSmoothing);

/*
 * Apply mixup for a given batch as in https://arxiv.org/abs/1710.09412
 */
std::pair<Tensor, Tensor> mixupBatch(
    const float lambda,
    const Tensor& input,
    const Tensor& target,
    const int numClasses,
    const float labelSmoothing);

/*
 * Apply cutmix as in https://arxiv.org/abs/1905.04899
 */
std::pair<Tensor, Tensor> cutmixBatch(
    const float lambda,
    const Tensor& input,
    const Tensor& target,
    const int numClasses,
    const float labelSmoothing);

// Same function signature as DataTransform but removes fl dep
using ImageTransform = std::function<Tensor(const Tensor&)>;

ImageTransform normalizeImage(
    const std::vector<float>& meanVec,
    const std::vector<float>& stdVec);

/*
 * Randomly resize the image between sizes
 * @param low
 * @param high
 * This transform helps to create scale invariance
 */
ImageTransform randomResizeTransform(const int low, const int high);

ImageTransform randomResizeCropTransform(
    const int resize,
    const float scaleLow,
    const float scaleHigh,
    const float ratioLow,
    const float ratioHigh);

/*
 * Randomly crop an image with target height of @param th and a target width of
 * @params tw
 */
ImageTransform randomCropTransform(const int th, const int tw);

/*
 * Resize the shortest edge of the image to size @param resize
 */
ImageTransform resizeTransform(const uint64_t resize);

/*
 * Take a center crop of an image so its size is @param size
 */
ImageTransform centerCropTransform(const int size);

/*
 * Flip an image horizontally with a probability @param p
 */
ImageTransform randomHorizontalFlipTransform(const float p = 0.5);

/*
 * Randomly erase.
 * The default parameters are defined as https://git.io/JY9R7
 *
 * @param[areaRatioMin] minimum area to erase
 * @param[areaRatioMax] maximum area to erase
 * @param[edgeRatioMin] minimum w/h ratio for the area to erase
 * @param[edgeRatioMax] maximum w/h ratio for the area to erase
 */
ImageTransform randomEraseTransform(
    const float p = 0.5,
    const float areaRatioMin = 0.02,
    const float areaRatioMax = 1. / 3.,
    const float edgeRatioMin = 0.3,
    const float edgeRatioMax = 10 / 3.);

/*
 * Randon Augmentation
 *
 * This implementation follows strictly the implementation used in
 * [DeiT](https://arxiv.org/abs/2012.12877). It's ource code can be found in
 * https://github.com/facebookresearch/deit.
 *
 * 15 augmentation transforms are randomly selected.
 *
 *
 * @param[p] the probablity of applying a certain transform, (1 - p) means the
 * probablity of skipping.
 * @param[n] number of transform functions to be applied on the input
 * @param[fillImg] filling values on the empty spots generated in some
 * transforms
 */
ImageTransform randomAugmentationDeitTransform(
    const float p = 0.5,
    const int n = 2,
    const Tensor& fillImg = Tensor());

/*
 * Utility method for composing multiple transform functions
 */
ImageTransform compose(std::vector<ImageTransform> transformfns);

} // namespace vision
} // namespace pkg
} // namespace fl
