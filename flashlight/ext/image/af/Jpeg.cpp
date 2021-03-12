/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/af/Jpeg.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace fl {
namespace ext {
namespace image {

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

void saveJpeg(const std::string& fp, const af::array& arr) {
  auto w = arr.dims(0);
  auto h = arr.dims(1);
  auto c = arr.dims(2);
  auto toOut = af::reorder(arr, 2, 0, 1).as(u8); // C x W x H

  // std::cout << w << ", " << h << ", " << c << std::endl;
  // std::cout << arr.elements() << std::endl;
  std::vector<uint8_t> vec(arr.elements());
  toOut.host(vec.data());

  stbi_write_jpg(fp.c_str(), w, h, c, vec.data(), 100);
}

} // namespace image
} // namespace ext
} // namespace fl
