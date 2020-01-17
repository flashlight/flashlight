#include <glob.h>
#include <fstream>

#include "flashlight/dataset/ImageDataset.h"

namespace {

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

} // namespace

namespace fl {

ImageDataset::ImageDataset(
    const std::string& sample_dir,
    const std::unordered_map<std::string, uint32_t>& labels,
    uint64_t resize)
    : resize_(resize) {
  auto files = glob(sample_dir + "/**/*.JPEG");
  if (files.size() == 0) {
    throw std::runtime_error("Could not file any files in " + sample_dir);
  }

  for (auto fp : files) {
    auto parent_path = fp.substr(0, fp.rfind("/"));
    auto label = parent_path.substr(parent_path.rfind("/") + 1);
    auto it = labels.find(label);
    if (it == labels.end()) {
      throw std::runtime_error("Could not find label:" + label + " in labels");
    } else {
      filepaths_.emplace_back(fp);
      labels_.emplace_back(labels.at(label));
    }
  }
}

std::vector<af::array> ImageDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  const std::string& filepath = filepaths_[idx];
  const af::array label = af::constant(labels_[idx], 1, 1, 1, 1, u64);
  af::array image = af::loadImage(filepath.c_str(), true);
  image = af::resize(image, resize_, resize_, AF_INTERP_BILINEAR);
  normalizeImage(image);
  return {image, label};
}

int64_t ImageDataset::size() const {
  return filepaths_.size();
}

std::unordered_map<std::string, uint32_t> ImageDataset::parseLabels(
    const std::string& label_file) {
  std::unordered_map<std::string, uint32_t> labels;
  std::ifstream fs(label_file);
  if (fs) {
    std::string line;
    for (uint64_t i = 0; std::getline(fs, line); i++) {
      auto it = line.find(",");
      if (it != std::string::npos) {
        std::string label = line.substr(0, it);
        labels[label] = i;
      } else {
        throw std::runtime_error("Invalid label format for line: " + line);
      }
    }
  } else {
    throw std::runtime_error("Could not open label file: " + label_file);
  }
  return labels;
}

void ImageDataset::normalizeImage(af::array& in) {
  in = in / 255.0f;
  in = (in - 0.5f) / 0.2f;
};

} // namespace fl
