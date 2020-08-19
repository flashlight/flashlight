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

#include "libraries/language/dictionary/Dictionary.h"

namespace fl {
namespace task {
namespace asr {

typedef std::unordered_map<int, af::dim4> DimsMap;
typedef std::unordered_map<int, std::vector<int>> TargetFeatMap;

struct W2lFeatureData {
  std::vector<float> input;
  TargetFeatMap targets;
  af::dim4 inputDims;
  DimsMap targetDims;
  std::vector<int> sampleIds;
  af::dim4 sampleIdsDims;
};

typedef std::unordered_map<int, std::vector<std::string>> TargetMap;
struct W2lLoaderData {
  std::vector<float> input;
  TargetMap targets;
  std::string sampleId;
};

W2lFeatureData featurize(
    const std::vector<W2lLoaderData>& data,
    const lib::DictionaryMap& dicts);

lib::FeatureParams defineSpeechFeatureParams();

int64_t getSpeechFeatureSize();
} // namespace asr
} // namespace task
} // namespace fl
