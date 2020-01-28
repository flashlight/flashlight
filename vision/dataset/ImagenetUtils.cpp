#include "vision/dataset/ImagenetUtils.h"

#include <algorithm>

#include "vision/dataset/ImageDataset.h"
#include "vision/dataset/Utils.h"
#include "vision/dataset/Transforms.h"
#include "flashlight/dataset/datasets.h"

namespace {

using namespace fl::cv::dataset;

using LabelLoader = Loader<uint64_t>;

LabelLoader labelLoader(std::vector<uint64_t> labels) {
  return LabelLoader(labels, [](uint64_t x) {
      std::vector<af::array> result {  af::constant(x, 1, 1, 1, 1, u64) };
      return result;
  });
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
      int size = labelMap.size();
      labelMap[label] = size;
    } 
    return labelMap[label];
  };
  std::vector<uint64_t> labels(filepaths.size());
  std::transform(filepaths.begin(), filepaths.end(), labels.begin(), getLabelTargets);
  return labels;
}


LabelLoader labelsFromSubDir(std::vector<std::string> fps) {
  std::vector<uint64_t> targets = labelTargets(fps);
  return labelLoader(targets);
}

} // namespace

namespace fl {
namespace cv {
namespace dataset {

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

std::shared_ptr<Dataset> imagenet(
    const std::string& img_dir,
    std::vector<Dataset::TransformFunction>& transformfns) {

  std::vector<std::string> filepaths = glob(img_dir + "/**/*.JPEG");

  auto images = std::make_shared<FilepathLoader>(jpegLoader(filepaths));
  // TransformDataset will apply each transform in a vector to the respective af::array
  // Thus, we need to `compose` all of the transforms so are each aplied
  std::vector<ImageTransform> transforms = { compose(transformfns) };
  auto transformed = std::make_shared<TransformDataset>(images, transforms);

  auto target_ds = std::make_shared<LabelLoader>(labelsFromSubDir(filepaths));
  return std::make_shared<MergeDataset>(MergeDataset({transformed, target_ds}));
}

} // namespace dataset
} // namespace cv
} // namespace flashlight
