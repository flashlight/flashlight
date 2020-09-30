/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Logger.h"

#include <thread>

#include <glog/logging.h>

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

using namespace fl::ext;
using namespace fl::lib;

namespace fl {
namespace app {
namespace asr {
std::pair<std::string, std::string> getStatus(
    TrainMeters& meters,
    int64_t epoch,
    int64_t nupdates,
    double lr,
    double lrcrit,
    bool verbose /* = false */,
    bool date /* = false */,
    const std::string& separator /* = " " */) {
  std::string errtype = "XER";
  errtype[0] = std::toupper(FLAGS_target[0]);
  std::string header, status;
  auto insertItem = [&](std::string key, std::string val) {
    if (verbose) {
      val = key + ": " + val;
    }
    header = header + (header.empty() ? "" : separator) + key;
    status = status + (status.empty() ? "" : separator) + val;
  };
  if (date) {
    insertItem("date", format("%s", getCurrentDate().c_str()));
    insertItem("time", format("%s", getCurrentTime().c_str()));
  }
  insertItem("epoch", format("%8d", epoch));
  insertItem("nupdates", format("%12d", nupdates));
  insertItem("lr", format("%4.6lf", lr));
  insertItem("lrcriterion", format("%4.6lf", lrcrit));

  int rt = meters.runtime.value();
  insertItem(
      "runtime",
      format("%02d:%02d:%02d", (rt / 60 / 60), (rt / 60) % 60, rt % 60));
  insertItem("bch(ms)", format("%.2f", meters.timer.value() * 1000));
  insertItem("smp(ms)", format("%.2f", meters.sampletimer.value() * 1000));
  insertItem("fwd(ms)", format("%.2f", meters.fwdtimer.value() * 1000));
  insertItem(
      "crit-fwd(ms)", format("%.2f", meters.critfwdtimer.value() * 1000));
  insertItem("bwd(ms)", format("%.2f", meters.bwdtimer.value() * 1000));
  insertItem("optim(ms)", format("%.2f", meters.optimtimer.value() * 1000));
  insertItem("loss", format("%10.5f", meters.train.loss.value()[0]));

  insertItem(
      "train-" + errtype, format("%5.2f", meters.train.tknEdit.value()[0]));
  insertItem("train-WER", format("%5.2f", meters.train.wrdEdit.value()[0]));
  for (auto& v : meters.valid) {
    insertItem(v.first + "-loss", format("%10.5f", v.second.loss.value()[0]));
    insertItem(
        v.first + "-" + errtype, format("%5.2f", v.second.tknEdit.value()[0]));
    insertItem(v.first + "-WER", format("%5.2f", v.second.wrdEdit.value()[0]));
  }
  auto stats = meters.stats.value();
  auto numsamples = std::max<int64_t>(stats[4], 1);
  auto isztotal = stats[0];
  auto tsztotal = stats[1];
  auto tszmax = stats[3];
  insertItem("avg-isz", format("%03d", isztotal / numsamples));
  insertItem("avg-tsz", format("%03d", tsztotal / numsamples));
  insertItem("max-tsz", format("%03d", tszmax));

  double audioProcSec = isztotal * FLAGS_batchsize;
  if (FLAGS_pow || FLAGS_mfcc || FLAGS_mfsc) {
    audioProcSec = audioProcSec * FLAGS_framestridems / 1000.0;
  } else {
    audioProcSec /= FLAGS_samplerate;
  }
  auto worldSize = fl::getWorldSize();
  double timeTakenSec = meters.timer.value() * numsamples / worldSize;

  insertItem("hrs", format("%7.2f", audioProcSec / 3600.0));
  insertItem(
      "thrpt(sec/sec)",
      timeTakenSec > 0.0 ? format("%.2f", audioProcSec / timeTakenSec) : "n/a");
  return {header, status};
}

void appendToLog(std::ofstream& logfile, const std::string& logstr) {
  auto write = [&]() {
    logfile.clear(); // reset flags
    logfile << logstr << std::endl;
    if (!logfile) {
      throw std::runtime_error("appending to log failed");
    }
  };
  retryWithBackoff(std::chrono::seconds(1), 1.0, 6, write);
}

af::array allreduceGet(SpeechStatMeter& mtr) {
  auto mtrVal0 = mtr.value();
  std::vector<long long> mtrVal(mtrVal0.begin(), mtrVal0.end());
  // Caveat: maxInputSz_, maxTargetSz_ would be approximate
  mtrVal[2] *= mtrVal[4];
  mtrVal[3] *= mtrVal[4];
  return af::array(mtrVal.size(), mtrVal.data());
}

void allreduceSet(SpeechStatMeter& mtr, af::array& val) {
  mtr.reset();
  // Caveat: maxInputSz_, maxTargetSz_ would be approximate
  auto valVec = afToVector<int64_t>(val);
  SpeechStats stats;
  auto denom = (valVec[4] == 0) ? 1 : valVec[4];
  stats.totalInputSz_ = valVec[0];
  stats.totalTargetSz_ = valVec[1];
  stats.maxInputSz_ = valVec[2] / denom;
  stats.maxTargetSz_ = valVec[3] / denom;
  stats.numSamples_ = valVec[4];
  mtr.add(stats);
}

void syncMeter(TrainMeters& mtrs) {
  fl::ext::syncMeter(mtrs.stats);
  fl::ext::syncMeter(mtrs.runtime);
  fl::ext::syncMeter(mtrs.timer);
  fl::ext::syncMeter(mtrs.fwdtimer);
  fl::ext::syncMeter(mtrs.critfwdtimer);
  fl::ext::syncMeter(mtrs.bwdtimer);
  fl::ext::syncMeter(mtrs.optimtimer);
  fl::ext::syncMeter(mtrs.train.tknEdit);
  fl::ext::syncMeter(mtrs.train.wrdEdit);
  fl::ext::syncMeter(mtrs.train.loss);
  for (auto& v : mtrs.valid) {
    fl::ext::syncMeter(v.second.tknEdit);
    fl::ext::syncMeter(v.second.wrdEdit);
    fl::ext::syncMeter(v.second.loss);
  }
}

} // namespace asr
} // namespace app
} // namespace fl
