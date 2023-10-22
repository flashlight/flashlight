/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/Imagenet.h"

#include <glob.h>
#include <algorithm>

#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/pkg/vision/dataset/Jpeg.h"
#include "flashlight/pkg/vision/dataset/LoaderDataset.h"
#include "flashlight/pkg/vision/dataset/Transforms.h"

using LabelLoader = fl::pkg::vision::LoaderDataset<uint64_t>;

namespace {

// TODO: test against std::filesystem::recursive_directory_iterator
std::vector<std::string> fileGlob(const std::string& pattern) {
  glob_t result;
  glob(pattern.c_str(), GLOB_TILDE, nullptr, &result);
  std::vector<std::string> ret;
  for (unsigned int i = 0; i < result.gl_pathc; ++i) {
    ret.emplace_back(result.gl_pathv[i]);
  }
  globfree(&result);
  return ret;
}

} // namespace
namespace fl::pkg::vision {

std::unordered_map<std::string, uint64_t> getImagenetLabels(
    const fs::path& labelFile) {
  std::unordered_map<std::string, uint64_t> labels;
  std::vector<std::string> lines;
  std::ifstream inFile(labelFile);
  if (!inFile) {
    throw std::invalid_argument(
        "fl::pkg::vision::getImagenetLabels given invalid labelFile path");
  }
  for (std::string str; std::getline(inFile, str);) {
    lines.emplace_back(str);
  }

  if (lines.empty()) {
    throw std::runtime_error(
        "In function imagenetLabels:  No lines in file:" + labelFile.string());
  }
  for (int i = 0; i < lines.size(); i++) {
    std::string line = lines[i];
    auto it = line.find(',');
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
    const fs::path& imgDir,
    const std::unordered_map<std::string, uint64_t>& labelMap,
    std::vector<Dataset::TransformFunction> transformfns) {
  std::vector<std::string> filepaths = fileGlob(imgDir.string() + "/**/*.JPEG");

  if (filepaths.empty()) {
    throw std::runtime_error(
        "No images were found in imagenet directory: " + imgDir.string());
  }

  // Create image dataset
  std::shared_ptr<Dataset> imageDataset =
      fl::pkg::vision::jpegLoader(filepaths);
  imageDataset = std::make_shared<TransformDataset>(imageDataset, transformfns);

  // Create labels from filepaths
  auto getLabelIdxs = [&labelMap](const std::string& s) -> uint64_t {
    std::string parentPath = s.substr(0, s.rfind('/'));
    std::string label = parentPath.substr(parentPath.rfind('/') + 1);
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

  auto labelDataset = std::make_shared<LabelLoader>(labels, [](uint64_t x) {
    std::vector<Tensor> result{fl::fromScalar(x, fl::dtype::u64)};
    return result;
  });
  return std::make_shared<MergeDataset>(
      MergeDataset({imageDataset, labelDataset}));
}

} // namespace fl
