/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <functional>

namespace fl {
namespace pkg {
namespace vision {

/*
 * Resizes the smallest length edge of an image to be resize while keeping
 * the aspect ratio
 */
af::array resizeSmallest(const af::array& in, const int resize);

/*
 * Resize both sides of image to be length
 * @param resize`
 * will change aspect ratio
 */
af::array resize(const af::array& in, const int resize);

/*
 * Crop image @param in, starting from position @param x and @param y
 * with a target width and height of @param w and @param h respectively
 */
af::array
crop(const af::array& in, const int x, const int y, const int w, const int h);

/*
 * Take a center crop of image @param in,
 * where both image sides with be of length @param size
 */
af::array centerCrop(const af::array& in, const int size);

/*
 * Rotate an image
 * @param theta to which degree (in radius) a image will rotate
 * @param fillImg filling values on the empty spots
 */
af::array
rotate(const af::array& input, const float theta, const af::array& fillImg);

/*
 * Skew an image on the first dimension
 * @param theta to which degree (in radius) a image will skew
 * @param fillImg filling values on the empty spots
 */
af::array
skewX(const af::array& input, const float theta, const af::array& fillImg);

/*
 * Skew an image on the second dimension
 * @param theta to which degree (in radius) a image will skew
 * @param fillImg filling values on the empty spots
 */
af::array
skewY(const af::array& input, const float theta, const af::array& fillImg);

/*
 * Translate an image on the first dimension
 * @param shift number of pixels a image will translate
 * @param fillImg filling values on the empty spots
 */
af::array
translateX(const af::array& input, const int shift, const af::array& fillImg);

/*
 * Translate an image on the second dimension
 * @param shift number of pixels a image will translate
 * @param fillImg filling values on the empty spots
 */
af::array
translateY(const af::array& input, const int shift, const af::array& fillImg);

/*
 * Enhance the color of an image
 * @param enhance to which extend the color will change.
 */
af::array colorEnhance(const af::array& input, const float enhance);

/*
 * Remaps the image so that the darkest pixel becomes black (0), and the
 * lightest becomes white (255).
 */
af::array autoContrast(const af::array& input);

/*
 * Enhance the contrast of an image
 * @param enhance to which extend the contrast will change.
 */
af::array contrastEnhance(const af::array& input, const float enhance);

/*
 * Enhance the brightness of an image
 * @param enhance to which extend the brightness will change.
 */
af::array brightnessEnhance(const af::array& input, const float enhance);

/*
 * Enhance the sharpness of an image
 * @param enhance to which extend the sharpness will change.
 */
af::array sharpnessEnhance(const af::array& input, const float enhance);

/*
 * Invert each pixel of the image
 */
af::array invert(const af::array& input);

/*
 * Invert all pixel values above a threshold.
 */
af::array solarize(const af::array& input, const float threshold);

/*
 * Increase all pixel values below a threshold.
 */
af::array solarizeAdd(
    const af::array& input,
    const float threshold,
    const float addValue);

/*
 * Applies a non-linear mapping to the input image, in order to create a uniform
 * distribution of grayscale values in the output image.
 */
af::array equalize(const af::array& input);

/*
 * Reduce the number of bits for each color channel.
 */
af::array posterize(const af::array& input, const int bitsToKeep);

/*
 * Transform a target array with label indices into a one-hot matrix
 */
af::array oneHot(
    const af::array& targets,
    const int numClasses,
    const float labelSmoothing);

/*
 * Apply mixup for a given batch as in https://arxiv.org/abs/1710.09412
 */
std::pair<af::array, af::array> mixupBatch(
    const float lambda,
    const af::array& input,
    const af::array& target,
    const int numClasses,
    const float labelSmoothing);

/*
 * Apply cutmix as in https://arxiv.org/abs/1905.04899
 */
std::pair<af::array, af::array> cutmixBatch(
    const float lambda,
    const af::array& input,
    const af::array& target,
    const int numClasses,
    const float labelSmoothing);

// Same function signature as DataTransform but removes fl dep
using ImageTransform = std::function<af::array(const af::array&)>;

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
    const af::array& fillImg = af::array());

/*
 * Utility method for composing multiple transform functions
 */
ImageTransform compose(std::vector<ImageTransform> transformfns);

} // namespace vision
} // namespace pkg
} // namespace fl
