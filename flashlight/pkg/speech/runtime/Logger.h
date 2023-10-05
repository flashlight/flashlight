/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>

#include "flashlight/fl/flashlight.h"
#include "flashlight/pkg/speech/runtime/SpeechStatMeter.h"

#define FL_LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

namespace fl {
namespace pkg {
namespace speech {
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
  fl::EditDistanceMeter wrdDstSlice;
  fl::EditDistanceMeter wrdDst;
  fl::EditDistanceMeter tknDstSlice;
  fl::EditDistanceMeter tknDst;
};

/*
 * Utility function to log results (learning rate, WER, TER, epoch, timing)
 * From gflags it uses FLAGS_batchsize, FLAGS_features_type
 * FLAGS_framestridems, FLAGS_samplerate
 */
std::string getLogString(
    TrainMeters& meters,
    const std::unordered_map<std::string, double>& dmErrs,
    int64_t epoch,
    int64_t nupdates,
    double lr,
    double lrcrit,
    double scaleFactor,
    const std::string& separator = " | ");

void appendToLog(std::ofstream& logfile, const std::string& logstr);

Tensor allreduceGet(SpeechStatMeter& mtr);
void allreduceSet(SpeechStatMeter& mtr, Tensor& val);

void syncMeter(TrainMeters& mtrs);
} // namespace speech
} // namespace pkg
} // namespace fl
