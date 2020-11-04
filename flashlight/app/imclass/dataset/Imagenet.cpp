/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/imclass/dataset/Imagenet.h"

#include <algorithm>

#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/ext/image/fl/dataset/Jpeg.h"
#include "flashlight/ext/image/fl/dataset/LoaderDataset.h"
#include "flashlight/lib/common/System.h"

using namespace fl::ext::image;

using LabelLoader = LoaderDataset<uint64_t>;

namespace fl {
namespace app {
namespace imclass {

std::unordered_map<std::string, uint64_t> getImagenetLabels(
    const std::string& labelFile) {
  std::unordered_map<std::string, uint64_t> labels;
  std::vector<std::string> lines = lib::getFileContent(labelFile);
  if (lines.empty()) {
    throw std::runtime_error(
        "In function imagenetLabels:  No lines in file:" + labelFile);
  }
  for (int i = 0; i < lines.size(); i++) {
    std::string line = lines[i];
    auto it = line.find(",");
    if (it != std::string::npos) {
      std::string label = line.substr(0, it);
      labels[label] = i;
    } else {
      throw std::runtime_error(
          "In function imagenetLabels: Invalid label format for line: " + line);
    }
  }
  return labels;
}

std::shared_ptr<Dataset> imagenetDataset(
    const std::string& imgDir,
    const std::unordered_map<std::string, uint64_t>& labelMap,
    std::vector<Dataset::TransformFunction> transformfns) {
  std::vector<std::string> filepaths = lib::fileGlob(imgDir + "/**/*.JPEG");
  if (filepaths.empty()) {
    throw std::runtime_error(
        "No images were found in imagenet directory: " + imgDir);
  }

  // Create image dataset
  std::shared_ptr<Dataset> imageDataset = jpegLoader(filepaths);
  imageDataset = std::make_shared<TransformDataset>(imageDataset, transformfns);

  // Create labels from filepaths
  auto getLabelIdxs = [&labelMap](const std::string& s) -> uint64_t {
    std::string parentPath = s.substr(0, s.rfind("/"));
    std::string label = parentPath.substr(parentPath.rfind("/") + 1);
    if (labelMap.find(label) != labelMap.end()) {
      return labelMap.at(label);
    } else {
      throw std::runtime_error("Label: " + label + " not found in label map");
    }
    return labelMap.at(label);
  };

  std::vector<uint64_t> labels(filepaths.size());
  std::transform(
      filepaths.begin(), filepaths.end(), labels.begin(), getLabelIdxs);

  auto labelDataset =
      std::make_shared<LabelLoader>(labels, [](uint64_t x) {
        std::vector<af::array> result{
            af::constant(x, 1, 1, 1, 1, af::dtype::u64)};
        return result;
      });
  return std::make_shared<MergeDataset>(
      MergeDataset({imageDataset, labelDataset}));
}

} // namespace imclass
} // namespace app
} // namespace fl
