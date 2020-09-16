#pragma once

#include "flashlight/dataset/Dataset.h"

namespace fl {
namespace cv {
namespace dataset {

using ImageTransform = Dataset::TransformFunction;


ImageTransform normalizeImage(
  const std::vector<float>& mean_,
  const std::vector<float>& std_);

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
ImageTransform horizontalFlipTransform(const float p = 0.5);

/*
 * Utility method for composing multiple transform functions
 */
ImageTransform compose(std::vector<ImageTransform>& transformfns);

} // transforms
} // cv
} // fg
