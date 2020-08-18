/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <unordered_map>

#include "FeatureTransforms.h"
#include "Sound.h"

#include "flashlight/lib/text/dictionary/Dictionary.h"

namespace fl {
namespace tasks {
namespace asr {

typedef std::unordered_map<int, af::dim4> DimsMap;
typedef std::unordered_map<int, std::vector<int>> TargetFeatMap;

struct FeatureData {
  std::vector<float> input;
  TargetFeatMap targets;
  af::dim4 inputDims;
  DimsMap targetDims;
  std::vector<int> sampleIds;
  af::dim4 sampleIdsDims;
};

typedef std::unordered_map<int, std::vector<std::string>> TargetMap;
struct LoaderData {
  std::vector<float> input;
  TargetMap targets;
  std::string sampleId;
};

FeatureData featurize(
    const std::vector<LoaderData>& data,
    const lib::text::DictionaryMap& dicts);

lib::audio::FeatureParams defineSpeechFeatureParams();

int64_t getSpeechFeatureSize();
} // namespace asr
} // namespace tasks
} // namespace fl
