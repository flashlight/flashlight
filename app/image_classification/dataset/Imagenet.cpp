/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/image_classification/dataset/Imagenet.h"

#include <algorithm>

#include "flashlight/dataset/datasets.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/ext/image/fl/dataset/Jpeg.h"
#include "flashlight/ext/image/fl/dataset/Loader.h"
#include "flashlight/lib/common/System.h"

namespace {

using namespace fl::ext::image;

using LabelLoader = Loader<uint64_t>;
using FilepathLoader = Loader<std::string>;

/*
 * Small function which creates a Dataset by loading scalars a vector of uint64_t
 */
LabelLoader labelLoader(std::vector<uint64_t> labels) {
  return LabelLoader(labels, [](uint64_t x) {
      std::vector<af::array> result {  af::constant(x, 1, 1, 1, 1, af::dtype::u64) };
      return result;
  });
}

/*
 * Assumes images are in a directory where the parent folder represents
 * thier class
 */
std::string labelFromFilePath(const std::string& fp) {
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
  auto getLabelTargets = [&labelMap](const std::string& s) -> uint32_t {
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


LabelLoader labelsFromSubDir(const std::vector<std::string>& fps) {
  std::vector<uint64_t> targets = labelTargets(fps);
  return labelLoader(targets);
}

} // namespace

namespace fl {
namespace app {
namespace image_classification {

std::unordered_map<std::string, uint32_t> imagenetLabels(
    const std::string& labelFile) {
  std::unordered_map<std::string, uint32_t> labels;
  std::vector<std::string> lines = lib::getFileContent(labelFile);
  if(lines.empty()) {
    throw std::runtime_error(
        "In function imagenetLabels:  No lines in file:" + labelFile
    );
  }
  std::ifstream fs(labelFile);
  for (int i = 0; i < lines.size(); i++) {
    std::string line = lines[i];
    auto it = line.find(",");
    if (it != std::string::npos) {
      std::string label = line.substr(0, it);
      labels[label] = i;
    } else {
      throw std::runtime_error(
          "In function imagenetLabels: Invalid label format for line: " + line
      );
    }
  }
  return labels;
}

std::shared_ptr<Dataset> imagenet(
    const std::string& imgDir,
    std::vector<Dataset::TransformFunction>& transformfns) {

  std::vector<std::string> filepaths = lib::fileGlob(imgDir + "/**/*.JPEG");
  if(filepaths.empty()) {
    throw std::runtime_error("No images were found in imagenet directory: " + imgDir);
  }

  auto images = jpegLoader(filepaths);
  // TransformDataset will apply each transform in a vector to the respective af::array
  // Thus, we need to `compose` all of the transforms so are each applied
  std::vector<ImageTransform> transforms = { compose(transformfns) };
  auto transformed = std::make_shared<TransformDataset>(images, transforms);
  auto target = std::make_shared<LabelLoader>(labelsFromSubDir(filepaths));
  return std::make_shared<MergeDataset>(MergeDataset({transformed, target}));
}

} // namespace image_classification
} // namespace app
} // namespace fl
