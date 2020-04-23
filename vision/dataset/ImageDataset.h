#pragma once

#include "vision/dataset/Utils.h"

namespace fl {
namespace cv {
namespace dataset {

using FilepathLoader = Loader<std::string>;

FilepathLoader jpegLoader(std::vector<std::string> fps);

static const uint64_t INPUT_IDX = 0;
static const uint64_t TARGET_IDX = 1;

af::array loadJpeg(const std::string& fp);

} // namespace dataset
} // namespace cv
} // namespace fl
