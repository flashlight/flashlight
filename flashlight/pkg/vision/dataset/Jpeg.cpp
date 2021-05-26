/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/Jpeg.h"

#include <memory>

#include "flashlight/pkg/vision/dataset/LoaderDataset.h"
#include "flashlight/fl/dataset/datasets.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fl {
namespace pkg {
namespace vision {

/*
 * Loads a jpeg from filepath fp. Note: It will automatically convert from any
 * number of channels to create an array with 3 channels
 */
af::array loadJpeg(const std::string& fp, int desiredNumberOfChannels) {
  int w, h, c;
  // STB image will automatically return desiredNumberOfChannels.
  // NB: c will be the original number of channels
  unsigned char* img =
      stbi_load(fp.c_str(), &w, &h, &c, desiredNumberOfChannels);
  if (img) {
    // Load array first as C X W X H, since stb has channel along first
    // dimension
    af::array result = af::array(desiredNumberOfChannels, w, h, img);
    stbi_image_free(img);
    // Then reorder to W X H X C
    return af::reorder(result, 1, 2, 0);
  } else {
    throw std::invalid_argument("Could not load from filepath" + fp);
  }
}

std::shared_ptr<Dataset> jpegLoader(std::vector<std::string> fps) {
  return std::make_shared<LoaderDataset<std::string>>(
      fps, [](const std::string& fp) {
        std::vector<af::array> result = {loadJpeg(fp)};
        return result;
      });
}

} // namespace vision
} // namespace pkg
} // namespace fl
