#pragma once

#include "vision/dataset/DataLoader.h"
#include "vision/dataset/Transforms.h"

namespace fl {
namespace cv {
namespace dataset {

using FilepathLoader = DataLoader<std::string>;

/*
 * Utility function for creating a Dataset from a list of filepaths
 */
FilepathLoader jpegLoader(std::vector<std::string> fps);

static const uint64_t INPUT_IDX = 0;
static const uint64_t TARGET_IDX = 1;

/*
 * Load a jpeg from a filepath into an af::array
 */
af::array loadJpeg(const std::string& fp);

} // namespace dataset
} // namespace cv
} // namespace fl
