#pragma once

#include <arrayfire.h>

namespace fl {
namespace app {
namespace objdet {

std::vector<af::array> crop(
    const std::vector<af::array>& in,
    int x,
    int y,
    int tw,
    int th);


} // namespace objdet
} // namespace app
} // namespace fl
