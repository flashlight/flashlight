/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Generic utilities which should not depend on ArrayFire / flashlight.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"

namespace fl {
namespace app {
namespace asr {

/* A series of vector to vector mapping operations */

std::vector<std::string> tknIdx2Ltr(
    const std::vector<int>& labels,
    const fl::lib::text::Dictionary& d,
    bool useWordPiece,
    const std::string& wordSep);

std::vector<std::string> tkn2Wrd(
    const std::vector<std::string>& input,
    const std::string& wordSep);

std::vector<std::string> wrdIdx2Wrd(
    const std::vector<int>& input,
    const fl::lib::text::Dictionary& wordDict);

std::vector<std::string> tknTarget2Ltr(
    std::vector<int> tokens,
    const fl::lib::text::Dictionary& tokenDict,
    const std::string& criterion,
    const std::string& surround,
    const bool eosTkn,
    const int replabel,
    const bool useWordPiece,
    const std::string& wordSep);

std::vector<std::string> tknPrediction2Ltr(
    std::vector<int> tokens,
    const fl::lib::text::Dictionary& tokenDict,
    const std::string& criterion,
    const std::string& surround,
    const bool eosTkn,
    const int replabel,
    const bool useWordPiece,
    const std::string& wordSep);

std::vector<int> tkn2Idx(
    const std::vector<std::string>& spelling,
    const fl::lib::text::Dictionary& tokenDict,
    int maxReps);

std::vector<int> validateIdx(std::vector<int> input, int unkIdx);

template <class T>
void remapLabels(
    std::vector<T>& labels,
    const fl::lib::text::Dictionary& dict,
    const std::string& surround,
    const bool eosTkn,
    const int replabel) {
  if (eosTkn) {
    int eosidx = dict.getIndex(kEosToken);
    while (!labels.empty() && labels.back() == eosidx) {
      labels.pop_back();
    }
  }
  if (replabel > 0) {
    labels = unpackReplabels(labels, dict, replabel);
  }
  auto trimLabels = [&labels](int idx) {
    if (!labels.empty() && labels.back() == idx) {
      labels.pop_back();
    }
    if (!labels.empty() && labels.front() == idx) {
      labels.erase(labels.begin());
    }
  };
  if (dict.contains(kSilToken)) {
    trimLabels(dict.getIndex(kSilToken));
  }
  if (!surround.empty()) {
    trimLabels(dict.getIndex(surround));
  }
};
} // namespace asr
} // namespace app
} // namespace fl
