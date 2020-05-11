#pragma once

#include "flashlight/dataset/datasets.h"
#include "vision/dataset/ImageDataset.h"

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
namespace cv {
namespace dataset {

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
 *
 * std::vector<ImageTransform> transforms = {
 *   cropTransform(224, 224),
 *   resizeTransform(224),
 *   normalizeImage({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225})
 * };
 * ds = imagenet(imagenet_base + "train", transforms);
 * auto sample = ds.get(0)
 * std::cout << sample[0].dims() << std::endl; // {224, 224, 3, 1}
 * std::cout << sample[1].dims() << std::endl; // {1, 1, 1, 1}
 *
 */
std::shared_ptr<Dataset> imagenet(
    const std::string& fp,
    std::vector<ImageTransform>& transformfns);

} // namespace dataset
} // namespace cv
} // namespace fl
