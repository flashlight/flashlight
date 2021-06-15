/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/af/Jpeg.h"

#include <memory>

#include "flashlight/ext/image/fl/dataset/LoaderDataset.h"
#include "flashlight/fl/dataset/datasets.h"

namespace fl {
namespace ext {
namespace image {

std::shared_ptr<Dataset> jpegLoader(std::vector<std::string> fps) {
  return std::make_shared<LoaderDataset<std::string>>(
      fps, [](const std::string& fp) {
        std::vector<af::array> result = {loadJpeg(fp)};
        return result;
      });
}

} // namespace image
} // namespace ext
} // namespace fl
