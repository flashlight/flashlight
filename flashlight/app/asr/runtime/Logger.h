/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include "flashlight/fl/flashlight.h"
#include "flashlight/app/asr/runtime/SpeechStatMeter.h"

#define FL_LOG_MASTER(lvl) FL_LOG_IF(lvl, (fl::getWorldRank() == 0))

namespace fl {
namespace app {
namespace asr {
struct DatasetMeters {
  fl::EditDistanceMeter tknEdit, wrdEdit;
  fl::AverageValueMeter loss;
};

struct TrainMeters {
  fl::TimeMeter runtime;
  fl::TimeMeter timer{true};
  fl::TimeMeter sampletimer{true};
  fl::TimeMeter fwdtimer{true}; // includes network + criterion time
  fl::TimeMeter critfwdtimer{true};
  fl::TimeMeter bwdtimer{true}; // includes network + criterion time
  fl::TimeMeter optimtimer{true};

  DatasetMeters train;
  std::map<std::string, DatasetMeters> valid;

  SpeechStatMeter stats;
};

struct TestMeters {
  fl::TimeMeter timer;
  fl::EditDistanceMeter werSlice;
  fl::EditDistanceMeter wer;
  fl::EditDistanceMeter lerSlice;
  fl::EditDistanceMeter ler;
};

std::pair<std::string, std::string> getStatus(
    TrainMeters& meters,
    int64_t epoch,
    int64_t nupdates,
    double lr,
    double lrcrit,
    bool verbose = false,
    bool date = false,
    const std::string& separator = " ");

void appendToLog(std::ofstream& logfile, const std::string& logstr);

af::array allreduceGet(SpeechStatMeter& mtr);
void allreduceSet(SpeechStatMeter& mtr, af::array& val);

void syncMeter(TrainMeters& mtrs);
} // namespace asr
} // namespace app
} // namespace fl
