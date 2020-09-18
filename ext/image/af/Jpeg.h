#pragma once

#include <arrayfire.h>

namespace fl {
namespace ext {
namespace image {

af::array loadJpeg(const std::string& fp);

} // namespace image
} // namespace ext
} // namespace fl
