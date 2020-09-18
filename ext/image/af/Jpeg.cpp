#include "flashlight/ext/image/af/Jpeg.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


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


//std::shared_ptr<Dataset> imageTransform(std::shared_ptr<Dataset> ds,
    //std::vector<ImageTransform>& transforms) {
  //std::vector<ImageTransform> composed = { compose(transforms) };
  //return std::make_shared<TransformDataset(ds, composed);
//}

} // namespace dataset
} // namespace cv
} // namespace fl
