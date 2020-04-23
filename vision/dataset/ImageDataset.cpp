#include "vision/dataset/ImageDataset.h"

#include <arrayfire.h>

#include <fstream>
#include <iostream>

#include "flashlight/dataset/Dataset.h"
#include "flashlight/dataset/MergeDataset.h"
#include "flashlight/dataset/TransformDataset.h"

#define STB_IMAGE_IMPLEMENTATION
#include "vision/dataset/stb_image.h"


namespace fl {
namespace cv {
namespace dataset {

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

FilepathLoader jpegLoader(std::vector<std::string> fps) {
  return FilepathLoader(fps, loadJpeg);
}


} // namespace dataset
} // namespace cv
} // namespace fl
