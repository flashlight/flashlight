#include "vision/dataset/ImageDataset.h"

#include <arrayfire.h>

#include <fstream>
#include <iostream>

#include "flashlight/dataset/Dataset.h"
#include "flashlight/dataset/MergeDataset.h"
#include "flashlight/dataset/TransformDataset.h"

#define STB_IMAGE_IMPLEMENTATION
#include "vision/dataset/stb_image.h"


namespace {

/*
 * Small generic utility class for loading data from a vector of type T into an
 * vector of arrayfire arrays
 */
template <typename T>
class Loader : public fl::Dataset {

public:
 using LoadFunc = std::function<af::array(const T&)>;

 Loader(const std::vector<T>& list, LoadFunc loadfn)
     : list_(list), loadfn_(loadfn) {}

 std::vector<af::array> get(const int64_t idx) const override {
   return {loadfn_(list_[idx])};
  }

  int64_t size() const override {
    return list_.size();
  }

  private:
  std::vector<T> list_;
  LoadFunc loadfn_;
};


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

af::array loadLabel(const uint64_t x) {
  return af::constant(x, 1, 1, 1, 1, u64);
}

}

namespace fl {


ImageDataset::ImageDataset(
    std::vector<std::string> filepaths,
    std::vector<uint64_t> labels,
    std::vector<TransformFunction>& transformfns
 ) : VisionDataset(transformfns) {
  // Create image loader and apply transforms
  // TransformDataset will apply each transform in a vector to the respective af::array
  // Thus, we need to `compose` all of the transforms so are each aplied
  auto images = std::make_shared<Loader<std::string>>(filepaths, loadJpeg);
  std::vector<TransformFunction> transforms = { compose(transformfns) };
  auto transformed = std::make_shared<TransformDataset>(images, transforms);

  // Create label loader
  auto targets = std::make_shared<Loader<uint64_t>>(labels, loadLabel);

  // Merge image and labels
  ds_ = std::make_shared<MergeDataset>(MergeDataset({transformed, targets}));
}

std::vector<af::array> ImageDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  return ds_->get(idx);
}

int64_t ImageDataset::size() const {
  return ds_->size();
}


} // namespace fl
