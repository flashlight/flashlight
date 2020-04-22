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

inline std::vector<std::string> glob(const std::string& pat) {
  glob_t result;
  glob(pat.c_str(), GLOB_TILDE, nullptr, &result);
  std::vector<std::string> ret;
  for (unsigned int i = 0; i < result.gl_pathc; ++i) {
    ret.push_back(std::string(result.gl_pathv[i]));
  }
  globfree(&result);
  return ret;
}

/*
 * Assumes images are in a directory where the parent folder represents 
 * thier class
 */
std::string labelFromFilePath(std::string fp) {
  auto parent_path = fp.substr(0, fp.rfind("/"));
  return parent_path.substr(parent_path.rfind("/") + 1);
}

/*
 * Given a vector of filepaths, and a dictionary of labels to labelIdx,
 * return a vector of label targets
 */
std::vector<uint64_t> labelTargets(
    const std::vector<std::string>& filepaths
    ) {
  std::unordered_map<std::string, uint32_t> labelMap;
  auto getLabelTargets = [&labelMap](const std::string& s) {
    const std::string label = labelFromFilePath(s);
    if (labelMap.find(label) == labelMap.end()) {
      labelMap[label] = labelMap.size();
    } 
    return labelMap[label];
  };
  std::vector<uint64_t> labels(filepaths.size());
  std::transform(filepaths.begin(), filepaths.end(), labels.begin(), getLabelTargets);
  return labels;
}

}

namespace fl {


ImageDataset::ImageDataset(
    const std::vector<std::string>& filepaths,
    const std::vector<TransformFunction>& transformfns
 ) : VisionDataset(transformfns), filepaths_(filepaths), labels_(labelTargets(filepaths_)) {
  if(filepaths_.size() != labels_.size()) {
    throw std::runtime_error("Filepaths size does not match label size");
  }
}

ImageDataset::ImageDataset(
    const std::string& dir,
    const std::vector<TransformFunction>& transformfns
 ) : ImageDataset(glob(dir + "/**/*.JPEG"), transformfns) {
}

std::vector<af::array> ImageDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  auto img = loadJpeg(filepaths_[idx]);
  for (auto fn : transformfns_) {
    img = fn(img);
  }
  auto target = loadLabel(labels_[idx]);
  return { img, target };
}

int64_t ImageDataset::size() const {
  return filepaths_.size();
}


} // namespace fl
