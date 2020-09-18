#pragma once

#include <arrayfire.h>

namespace fl {
namespace cv {
namespace dataset {

af::array loadJpeg(const std::string& fp);

} // namespace dataset
} // namespace cv
} // namespace fl
