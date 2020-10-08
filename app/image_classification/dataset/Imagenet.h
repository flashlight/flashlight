/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <string>

#include "flashlight/ext/image/af/Jpeg.h"
#include "flashlight/flashlight/dataset/datasets.h"
/**
 * Utilities for creating an ImageDataset with imagenet data
 * The jpegs must be placed in subdirectories representing their class in a
 * similar fashion to imagenet.
 *
 * For example
 * train/
 * >> n01440764/
 * >>>> n01440764_10026.JPEG
 * >>>> n01440764_10027.JPEG
 * val/
 * >> n01440764
 * >>>> ILSVRC2012_val_00000293.JPEG
 * >>>> ILSVRC2012_val_00002138.JPEG
 * ...
 * labels.txt
 * ....
 * n01440764,tench
 * n01443537,goldfish
 * n01484850,great white shark
 * n01491361,tiger shark
 * .....
 *
 */
namespace fl {
namespace app {
namespace image_classification {

/* Given the path to the imagenet labels file labels.txt,
 * create a map with a unique id for each label that can be used for training
 */
std::unordered_map<std::string, uint32_t> imagenetLabels(
    const std::string& label_file);

/*
 * Creates an `ImageDataset` by globbing for images in
 * @param[fp] and determines their labels using @params[labelIdxs].
 * \code{.cpp}
 * std::string imagenet_base = "/data/imagenet/";
 * auto labels = imagenetLabels(imagenet_base + "labels.txt");
 *
 * std::vector<Dataset::TransformFunction> transforms = {
 *   ImageDataset::cropTransform(224, 224),
 *   ImageDataset::resizeTransform(224),
 *   ImageDataset::normalizeImage({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})
 * };
 * ds = imagenetDataset(imagenet_base + "train", labels, transforms);
 * auto sample = ds.get(0)
 * std::cout << sample[0].dims() << std::endl; // {224, 224, 3, 1}
 * std::cout << sample[1].dims() << std::endl; // {1, 1, 1, 1}
 *
 */
std::shared_ptr<Dataset> imagenet(
    const std::string& fp,
    std::vector<Dataset::TransformFunction>& transformfns);

constexpr uint64_t kImagenetInputIdx = 0;
constexpr uint64_t kImagenetTargetIdx = 1;

} // namespace image_classification
} // namespace app
} // namespace fl
