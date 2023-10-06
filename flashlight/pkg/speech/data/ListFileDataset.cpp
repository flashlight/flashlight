/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/data/ListFileDataset.h"

#include "flashlight/lib/text/String.h"
#include "flashlight/pkg/speech/data/Sound.h"

using namespace fl::lib;

namespace {
constexpr const size_t kIdIdx = 0;
constexpr const size_t kInIdx = 1;
constexpr const size_t kSzIdx = 2;
constexpr const size_t kTgtIdx = 3;
constexpr const size_t kNumCols = 4;

} // namespace

namespace fl::pkg::speech {
ListFileDataset::ListFileDataset(
    const std::string& filename,
    const DataTransformFunction& inFeatFunc /* = nullptr */,
    const DataTransformFunction& tgtFeatFunc /* = nullptr */,
    const DataTransformFunction& wrdFeatFunc /* = nullptr */)
    : inFeatFunc_(inFeatFunc),
      tgtFeatFunc_(tgtFeatFunc),
      wrdFeatFunc_(wrdFeatFunc),
      numRows_(0) {
  std::ifstream inFile(filename);
  if (!inFile) {
    throw std::invalid_argument("Unable to open file -" + filename);
  }
  std::string line;
  while (std::getline(inFile, line)) {
    if (line.empty()) {
      continue;
    }
    auto splits = splitOnWhitespace(line, true);
    if (splits.size() < 3) {
      throw std::runtime_error(
          "File " + filename +
          " has invalid columns in line (expected 3 columns at least): " +
          line);
    }

    ids_.emplace_back(std::move(splits[kIdIdx]));
    inputs_.emplace_back(std::move(splits[kInIdx]));
    inputSizes_.emplace_back(std::stof(splits[kSzIdx]));
    targets_.emplace_back(fl::lib::join(
        " ", std::vector<std::string>(splits.begin() + kTgtIdx, splits.end())));
    ++numRows_;
  }
  inFile.close();
  targetSizesCache_.resize(inputSizes_.size(), -1);
}

int64_t ListFileDataset::size() const {
  return numRows_;
}

std::vector<Tensor> ListFileDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);

  auto audio = loadAudio(inputs_[idx]); // channels x time
  Tensor input;
  if (inFeatFunc_) {
    input = inFeatFunc_(
        static_cast<void*>(audio.first.data()), audio.second, fl::dtype::f32);
  } else {
    input = Tensor::fromBuffer(
        {audio.second}, audio.first.data(), MemoryLocation::Host);
  }

  Tensor target;
  if (tgtFeatFunc_) {
    std::vector<char> curTarget(targets_[idx].begin(), targets_[idx].end());
    target = tgtFeatFunc_(
        static_cast<void*>(curTarget.data()),
        {static_cast<Dim>(curTarget.size())},
        fl::dtype::b8);
  }
  targetSizesCache_[idx] = target.elements();

  Tensor words;
  if (wrdFeatFunc_) {
    std::vector<char> curTarget(targets_[idx].begin(), targets_[idx].end());
    words = wrdFeatFunc_(
        static_cast<void*>(curTarget.data()),
        {static_cast<Dim>(curTarget.size())},
        fl::dtype::b8);
  }

  Tensor sampleIdx = Tensor::fromBuffer(
      {static_cast<long long>(ids_[idx].length())},
      const_cast<char*>(ids_[idx].data()), // fix me post C++-17?
      MemoryLocation::Host);
  Tensor samplePath = Tensor::fromBuffer(
      {static_cast<long long>(inputs_[idx].length())},
      inputs_[idx].data(),
      MemoryLocation::Host);
  Tensor sampleDuration =
      Tensor::fromBuffer({1}, inputSizes_.data() + idx, MemoryLocation::Host);
  Tensor sampleTargetSize = fl::full({1}, float(target.elements()));

  return {
      input,
      target,
      words,
      sampleIdx,
      samplePath,
      sampleDuration,
      sampleTargetSize};
}

std::pair<std::vector<float>, Shape> ListFileDataset::loadAudio(
    const std::string& handle) const {
  auto info = loadSoundInfo(handle.c_str());
  return {loadSound<float>(handle.c_str()), {info.channels, info.frames}};
}

float ListFileDataset::getInputSize(const int64_t idx) const {
  checkIndexBounds(idx);
  return inputSizes_[idx];
}

int64_t ListFileDataset::getTargetSize(const int64_t idx) const {
  checkIndexBounds(idx);
  if (targetSizesCache_[idx] >= 0) {
    return targetSizesCache_[idx];
  }
  if (!tgtFeatFunc_) {
    return 0;
  }
  std::vector<char> curTarget(targets_[idx].begin(), targets_[idx].end());
  auto tgtSize = tgtFeatFunc_(
                     static_cast<void*>(curTarget.data()),
                     {static_cast<long long>(curTarget.size())},
                     fl::dtype::b8)
                     .elements();
  targetSizesCache_[idx] = tgtSize;
  return tgtSize;
}

} // namespace fl
