#include <algorithm>
#include <fstream>
#include <glob.h>

#include "flashlight/dataset/ImagenetUtils.h"

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

/*
 * Assumes images are in a directory where the parent folder represents 
 * thier class
 */
std::string labelFromFilePath(std::string fp) {
  auto parent_path = fp.substr(0, fp.rfind("/"));
  return parent_path.substr(parent_path.rfind("/") + 1);
}

std::vector<uint64_t> labelTargets(
    std::vector<std::string>& filepaths,
    std::unordered_map<std::string, uint32_t>& labelMap
    ) {
  auto getLabelTargets = [&labelMap](const std::string& s) {
    const std::string label = labelFromFilePath(s);
    return labelMap.at(label);
  };
  std::vector<uint64_t> labels(filepaths.size());
  std::transform(filepaths.begin(), filepaths.end(), labels.begin(), getLabelTargets);
  return labels;
}


} // namespace

namespace fl {

std::unordered_map<std::string, uint32_t> imagenetLabels(
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

ImageDataset imagenetDataset(
    const std::string& fp,
    std::unordered_map<std::string, uint32_t>& labelIdxs,
    std::vector<Dataset::TransformFunction>& transformfns) {
  auto filepaths = glob(fp + "/**/*.JPEG");
  if (filepaths.size() == 0) {
    throw std::runtime_error("Could not file any files in " + fp);
  }
  auto labels = labelTargets(filepaths, labelIdxs);
  return ImageDataset(filepaths, labels, transformfns);
}
} // namespace fl
