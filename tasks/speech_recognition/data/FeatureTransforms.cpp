/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "FeatureTransforms.h"

#include <algorithm>
#include <stdexcept>

#include "libraries/audio/feature/Mfcc.h"
#include "libraries/audio/feature/Mfsc.h"
#include "libraries/audio/feature/PowerSpectrum.h"
#include "libraries/common/String.h"

namespace {

// Input: B x inRow x inCol (Row Major), Output: B x inCol x inRow (Row Major)
template <typename T>
std::vector<T> transpose2d(
    const std::vector<T>& in,
    int64_t inRow,
    int64_t inCol,
    int64_t inBatch = 1) {
  if (in.size() != inRow * inCol * inBatch) {
    throw std::invalid_argument("Invalid input size");
  }
  std::vector<T> out(in.size());
  for (size_t b = 0; b < inBatch; ++b) {
    int64_t start = b * inRow * inCol;
    for (size_t c = 0; c < inCol; ++c) {
      for (size_t r = 0; r < inRow; ++r) {
        out[start + c * inRow + r] = in[start + r * inCol + c];
      }
    }
  }
  return out;
}

}

namespace w2l {

fl::Dataset::DataTransformFunction inputFeatures(const FeatureParams& params) {
  return [params](void* data, af::dim4 dims, af::dtype type) {
    if (type != af::dtype::f32) {
      throw std::invalid_argument("Invalid input type");
    }
    if (dims[2] != 1 || dims[3] != 1) {
      throw std::invalid_argument("Invalid input dims");
    }
    auto channels = dims[0];
    std::vector<float> input(dims.elements());
    std::copy_n(static_cast<const float*>(data), input.size(), input.data());

    if (channels > 1) {
      input = transpose2d(input, dims[1], channels);
    }

    static Mfsc mfsc(params);
    auto output = mfsc.apply(input);
    auto nFeat = params.mfscFeatSz();
    auto T = output.size() / (nFeat * channels);

    output = transpose2d(output, nFeat, T, channels);
    return af::array(T, nFeat, channels, output.data());
  };
}

// target
fl::Dataset::DataTransformFunction targetFeatures(
    const Dictionary& dict,
    const LexiconMap& lexicon,
    int wordSeperator,
    bool surround /* = true */) {
  return [dict, lexicon, wordSeperator, surround](
             void* data, af::dim4 dims, af::dtype type) {
    std::string transcript((char*)data, (char*)data + dims[0]);
    auto words = splitOnWhitespace(transcript);
    std::vector<int> tokens;
    if (surround && wordSeperator >= 0) {
      tokens.emplace_back(wordSeperator);
    }
    for (const auto& word : words) {
      auto iter = lexicon.find(word);
      if (iter != lexicon.end()) {
        auto wTkns = dict.mapEntriesToIndices(iter->second[0]);
        tokens.insert(tokens.end(), wTkns.begin(), wTkns.end());
      }
      if (wordSeperator >= 0) {
        tokens.emplace_back(wordSeperator);
      }
    }
    if (!surround && !tokens.empty()) {
      tokens.pop_back();
    }
    return af::array(tokens.size(), tokens.data());
  };
}
} // namespace w2l
