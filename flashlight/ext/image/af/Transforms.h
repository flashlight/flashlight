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
namespace ext {
namespace image {

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

} // namespace image
} // namespace ext
} // namespace fl
