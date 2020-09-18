#pragma once

#include <memory>

#include "flashlight/dataset/datasets.h"

namespace fl {
namespace ext {
namespace image {

std::shared_ptr<Dataset> jpegLoader(std::vector<std::string> fps);

}
}
}
