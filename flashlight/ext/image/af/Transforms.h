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
 * Utility method for composing multiple transform functions
 */
ImageTransform compose(std::vector<ImageTransform> transformfns);

} // namespace image
} // namespace ext
} // namespace fl
