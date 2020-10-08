/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Jpeg.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


namespace fl {
namespace ext {
namespace image {

/*
 * Loads a jpeg from filepath fp. Note: It will automatically convert from any
 * numnber of channels to create an array with 3 channels
 */
af::array loadJpeg(const std::string& fp) {
  int w, h, c;
  // STB image will automatically return desired_no_channels.
  // NB: c will be the original number of channels
  int desired_no_channels = 3;
  unsigned char *img = stbi_load(fp.c_str(), &w, &h, &c, desired_no_channels);
  if (img) {
    af::array result = af::array(desired_no_channels, w, h, img);
    stbi_image_free(img);
    return af::reorder(result, 1, 2, 0);
  } else {
    throw std::invalid_argument("Could not load from filepath" + fp);
  }
}

} // namespace image
} // namespace ext
} // namespace fl
