#pragma once

#include "vision/dataset/Utils.h"
#include "vision/dataset/Transforms.h"

namespace fl {
namespace cv {
namespace dataset {

using FilepathLoader = Loader<std::string>;

FilepathLoader jpegLoader(std::vector<std::string> fps);

static const uint64_t INPUT_IDX = 0;
static const uint64_t TARGET_IDX = 1;

af::array loadJpeg(const std::string& fp);

// TransformDataset will apply each transform in a vector to the respective af::array
// Thus, we need to `compose` all of the transforms so are each aplied
//std::shared_ptr<Dataset> imageTransform(
    //std::shared_ptr<Dataset> ds,
    //std::vector<ImageTransform>& transforms);

} // namespace dataset
} // namespace cv
} // namespace fl
