/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

#include "flashlight/tasks/speech_recognition/criterion/criterion.h"
#include "flashlight/tasks/speech_recognition/data/W2lDataset.h"
#include "flashlight/libraries/common/Dictionary.h"
#include "recipes/models/local_prior_match/src/runtime/Logging.h"
#include "flashlight/tasks/speech_recognition/runtime/runtime.h"

namespace w2l {
void evalOutput(
    const af::array& op,
    const af::array& target,
    std::map<std::string, fl::EditDistanceMeter>& mtr,
    const Dictionary& tgtDict,
    std::shared_ptr<SequenceCriterion> criterion);

void evalDataset(
    std::shared_ptr<fl::Module> ntwrk,
    std::shared_ptr<SequenceCriterion> crit,
    std::shared_ptr<W2lDataset> testds,
    SSLDatasetMeters& mtrs,
    const Dictionary& dict);

void runEval(
    std::shared_ptr<fl::Module> network,
    std::shared_ptr<SequenceCriterion> criterion,
    const std::unordered_map<std::string, std::shared_ptr<W2lDataset>>& ds,
    SSLTrainMeters& meters,
    const Dictionary& dict);

} // namespace w2l
