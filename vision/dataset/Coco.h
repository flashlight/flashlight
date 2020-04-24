#pragma once

#include "vision/dataset/ImageDataset.h"
#include "vision/dataset/Transforms.h"
#include "flashlight/dataset/datasets.h"

namespace fl {
namespace cv {
namespace dataset {

std::shared_ptr<Dataset> coco(
    const std::string& list_file,
    std::vector<ImageTransform>& transformfns);

} // namespace dataset
} // namespace cv
} // namespace flashlight
