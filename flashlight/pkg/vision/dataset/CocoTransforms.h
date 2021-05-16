/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>

#include "flashlight/pkg/vision/dataset/TransformAllDataset.h"

namespace fl {
namespace app {
namespace objdet {

enum DatasetIndices {
  ImageIdx = 0,
  TargetSizeIdx = 1,
  ImageIdIdx = 2,
  OriginalSizeIdx = 3,
  BboxesIdx = 4,
  ClassesIdx = 5
};

/*
 * Crop the image and translate bounding boxes accordingly
 * @param x is the starting position of crop along the first dimension
 * @param y is the starting position of crop along the second dimension
 * @param tw is the target width
 * @param ty is the target height
 * This function will remove bounding boxes which do not exist within the crop
 */
std::vector<af::array>
crop(const std::vector<af::array>& in, int x, int y, int tw, int th);

/*
 * Flip the image horizontally and adjust the bounding boxes acordingly
 * @param in vector of input arrays
 */
std::vector<af::array> hflip(const std::vector<af::array>& in);

/*
 * "normalize" the bounding boxes
 * @param in input arrays
 * adjust bounding boxes from bottom left and top right coordinates to center
 * x,y and width and height and then divide by total image width and height
 */
std::vector<af::array> normalize(const std::vector<af::array>& in);

/*
 * Randomly resize image and bounding boxes from @param inputs, where shortest
 * so shortest length side is of size @param size, but clip so that longest
 * side is shorter than @param maxsize.
 * Adjust bboxes accordingly.
 */
std::vector<af::array>
randomResize(std::vector<af::array> inputs, int size, int maxsize);

/*
 * Returns a function that "Normalizes" bounding boxes so that they represent
 * center coordintates and height and width ratios of the entire image.
 * Also normalize images by @param meanVector and @param stdVector
 */
TransformAllFunction Normalize(
    std::vector<float> meanVector = {0.485, 0.456, 0.406},
    std::vector<float> stdVector = {0.229, 0.224, 0.225});

/*
 * Returns a `TransformAllFunction` which randomly selects from @param fns
 * and call it on imput data
 */
TransformAllFunction randomSelect(std::vector<TransformAllFunction> fns);

/*
 * Returns a `TransformAllFunction` which randomly resizes image and bounding
 * boxes between @param minSize and @param maxSize
 */
TransformAllFunction randomSizeCrop(int minSize, int maxSize);

/*
 * Returns a `TransformAllFunction` which randomly resizes images from a choice
 * in @param sizes, and ensures longest side of image is less than @param
 * maxsize
 */
TransformAllFunction randomResize(std::vector<int> sizes, int maxsize);

/*
 * Returns a `TransformAllFunction` which random flips image and bounding
 * boxes with a probility of @param p
 */
TransformAllFunction randomHorizontalFlip(float p);

/*
 * Returns a `TransformAllFunction` which calls each @param fns on
 * the input data
 */
TransformAllFunction compose(std::vector<TransformAllFunction> fns);

} // namespace objdet
} // namespace app
} // namespace fl
