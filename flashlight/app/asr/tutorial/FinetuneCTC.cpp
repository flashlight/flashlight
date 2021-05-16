/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/app/asr/augmentation/SoundEffectConfig.h"
#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/criterion/criterion.h"
#include "flashlight/app/asr/data/FeatureTransforms.h"
#include "flashlight/app/asr/data/Utils.h"
#include "flashlight/app/asr/decoder/TranscriptionUtils.h"
#include "flashlight/app/asr/runtime/runtime.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/common/SequentialBuilder.h"
#include "flashlight/ext/common/Serializer.h"
#include "flashlight/pkg/runtime/Runtime.h"
#include "flashlight/ext/plugin/ModulePlugin.h"
#include "flashlight/fl/contrib/contrib.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"

using fl::ext::Serializer;
using fl::ext::afToVector;
using fl::app::getRunFile;
using fl::lib::fileExists;
using fl::lib::format;
using fl::lib::getCurrentDate;
using fl::lib::join;
using fl::lib::pathsConcat;

using namespace fl::app::asr;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage("Usage: \n " + exec + " [model] [flags]");

  fl::init();

  /* ===================== Parse Options ===================== */
  int runIdx = 1; // current #runs in this path
  std::string reloadPath = argv[1]; // path to model to reload
  std::unordered_map<std::string, std::string> cfg;
  int64_t startEpoch = 0;
  int64_t startUpdate = 0;
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }
  std::string version;
  Serializer::load(reloadPath, version, cfg);
  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "Invalid config loaded from " << reloadPath;
  }

  LOG(INFO) << "Reading flags from config file " << reloadPath;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

  if (argc > 3) {
    LOG(INFO) << "Parsing command line flags";
    gflags::ParseCommandLineFlags(&argc, &argv, false);
  }

  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file" << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  std::string runPath = FLAGS_rundir;
  handleDeprecatedFlags();

  fl::setSeed(FLAGS_seed);
  fl::DynamicBenchmark::setBenchmarkMode(FLAGS_fl_benchmark_mode);

  std::shared_ptr<fl::Reducer> reducer = nullptr;
  if (FLAGS_enable_distributed) {
    fl::ext::initDistributed(
        FLAGS_world_rank,
        FLAGS_world_size,
        FLAGS_max_devices_per_node,
        FLAGS_rndv_filepath);
    reducer = std::make_shared<fl::CoalescingReducer>(
        1.0 / fl::getWorldSize(), true, true);
  }

  int worldRank = fl::getWorldRank();
  int worldSize = fl::getWorldSize();
  bool isMaster = (worldRank == 0);

  FL_LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");
  FL_LOG_MASTER(INFO) << "Experiment path: " << runPath;
  FL_LOG_MASTER(INFO) << "Experiment runidx: " << runIdx;

  // flashlight optim mode
  auto flOptimLevel = FLAGS_fl_optim_mode.empty()
      ? fl::OptimLevel::DEFAULT
      : fl::OptimMode::toOptimLevel(FLAGS_fl_optim_mode);
  fl::OptimMode::get().setOptimLevel(flOptimLevel);
  if (FLAGS_fl_amp_use_mixed_precision) {
    // Only set the optim mode to O1 if it was left empty
    LOG(INFO) << "Mixed precision training enabled. Will perform loss scaling.";
    if (FLAGS_fl_optim_mode.empty()) {
      LOG(INFO) << "Mixed precision training enabled with no "
                   "optim mode specified - setting optim mode to O1.";
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
    }
  }

  std::unordered_map<std::string, std::string> config = {
      {kProgramName, exec},
      {kCommandLine, join(" ", argvs)},
      {kGflags, serializeGflags()},
      // extra goodies
      {kUserName, fl::lib::getEnvVar("USER")},
      {kHostName, fl::lib::getEnvVar("HOSTNAME")},
      {kTimestamp, getCurrentDate() + ", " + getCurrentDate()},
      {kRunIdx, std::to_string(runIdx)},
      {kRunPath, runPath}};

  auto validTagSets = parseValidSets(FLAGS_valid);

  /* ===================== Create Dictionary & Lexicon ===================== */
  auto dictPath = FLAGS_tokens;
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error(
        "Invalid dictionary filepath specified with --tokensdir and --tokens: \"" +
        dictPath + "\"");
  }
  fl::lib::text::Dictionary tokenDict(dictPath);
  if (FLAGS_criterion != kCtcCriterion) {
    LOG(FATAL) << "Finetune binary works only with ctc criterion.";
  }
  // ctc expects the blank label last
  tokenDict.addEntry(kBlankToken);

  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  fl::lib::text::Dictionary wordDict;
  fl::lib::text::LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = fl::lib::text::loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = fl::lib::text::createWordDict(lexicon);
    LOG(INFO) << "Number of words: " << wordDict.indexSize();
  }

  /* ===================== Create Dataset ===================== */
  fl::lib::audio::FeatureParams featParams(
      FLAGS_samplerate,
      FLAGS_framesizems,
      FLAGS_framestridems,
      FLAGS_filterbanks,
      FLAGS_lowfreqfilterbank,
      FLAGS_highfreqfilterbank,
      FLAGS_mfcccoeffs,
      kLifterParam /* lifterparam */,
      FLAGS_devwin /* delta window */,
      FLAGS_devwin /* delta-delta window */);
  featParams.useEnergy = false;
  featParams.usePower = false;
  featParams.zeroMeanFrame = false;
  auto featureRes =
      getFeatureType(FLAGS_features_type, FLAGS_channels, featParams);
  int numFeatures = featureRes.first;
  FeatureType featType = featureRes.second;

  TargetGenerationConfig targetGenConfig(
      FLAGS_wordseparator,
      FLAGS_sampletarget,
      FLAGS_criterion,
      FLAGS_surround,
      false /* isSeq2seqCrit */,
      FLAGS_replabel,
      true /* skip unk */,
      FLAGS_usewordpiece /* fallback2LetterWordSepLeft */,
      !FLAGS_usewordpiece /* fallback2LetterWordSepLeft */);

  const auto sfxConf = (FLAGS_sfx_config.empty())
      ? std::vector<sfx::SoundEffectConfig>()
      : sfx::readSoundEffectConfigFile(FLAGS_sfx_config);

  auto inputTransform = inputFeatures(
      featParams,
      featType,
      {FLAGS_localnrmlleftctx, FLAGS_localnrmlrightctx},
      sfxConf);
  auto targetTransform = targetFeatures(tokenDict, lexicon, targetGenConfig);
  auto wordTransform = wordFeatures(wordDict);
  int targetpadVal = kTargetPadValue;
  int wordpadVal = kTargetPadValue;

  std::vector<std::string> trainSplits = fl::lib::split(",", FLAGS_train, true);
  auto trainds = createDataset(
      trainSplits,
      FLAGS_datadir,
      FLAGS_batchsize,
      inputTransform,
      targetTransform,
      wordTransform,
      std::make_tuple(0, targetpadVal, wordpadVal),
      worldRank,
      worldSize,
      false // allowEmpty
  );

  std::map<std::string, std::shared_ptr<fl::Dataset>> validds;
  int64_t validBatchSize =
      FLAGS_validbatchsize == -1 ? FLAGS_batchsize : FLAGS_validbatchsize;
  for (const auto& s : validTagSets) {
    validds[s.first] = createDataset(
        {s.second},
        FLAGS_datadir,
        validBatchSize,
        inputTransform,
        targetTransform,
        wordTransform,
        std::make_tuple(0, targetpadVal, wordpadVal),
        worldRank,
        worldSize,
        true // allowEmpty
    );
  }

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  std::shared_ptr<fl::Module> network;
  auto archfile = FLAGS_arch;
  FL_LOG_MASTER(INFO) << "Loading architecture file from " << archfile;
  // Encoder network, works on audio
  if (fl::lib::endsWith(archfile, ".so")) {
    network = fl::ext::ModulePlugin(archfile).arch(numFeatures, numClasses);
  } else {
    network = fl::ext::buildSequentialModule(archfile, numFeatures, numClasses);
  }

  std::shared_ptr<fl::Module> forkingNetwork;
  Serializer::load(reloadPath, version, cfg, forkingNetwork);
  if (version != FL_APP_ASR_VERSION) {
    LOG(WARNING) << "Model version " << version << " and code version "
                 << FL_APP_ASR_VERSION;
  }
  // override params
  if (forkingNetwork->params().size() != network->params().size()) {
    LOG(FATAL)
        << "Mismatch in # parameters for the model specificied by archfile and forking model.";
  }
  for (int i = 0; i < forkingNetwork->params().size(); ++i) {
    if (network->param(i).dims() != forkingNetwork->param(i).dims()) {
      LOG(FATAL) << "Mismatch in parameter dims for position " << i
                 << ". Expected: " << network->param(i).dims()
                 << " Got: " << forkingNetwork->param(i).dims();
    }

    network->setParams(forkingNetwork->param(i), i);
  }

  FL_LOG_MASTER(INFO) << "[Network] " << network->prettyString();
  FL_LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network) << "]";

  auto scalemode = getCriterionScaleMode(FLAGS_onorm, FLAGS_sqnorm);
  std::shared_ptr<SequenceCriterion> criterion =
      std::make_shared<CTCLoss>(scalemode);
  FL_LOG_MASTER(INFO) << "[Criterion] " << criterion->prettyString();

  std::shared_ptr<fl::FirstOrderOptimizer> netoptim = initOptimizer(
      {network}, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
  FL_LOG_MASTER(INFO) << "[Network Optimizer] " << netoptim->prettyString();

  /* ===================== Meters ===================== */
  TrainMeters meters;
  for (const auto& s : validTagSets) {
    meters.valid[s.first] = DatasetMeters();
  }

  // best perf so far on valid datasets
  std::unordered_map<std::string, double> validminerrs;
  for (const auto& s : validTagSets) {
    validminerrs[s.first] = DBL_MAX;
  }

  /* ===================== Logging ===================== */
  std::ofstream logFile;
  if (isMaster) {
    fl::lib::dirCreate(runPath);
    logFile.open(getRunFile("log", runIdx, runPath));
    if (!logFile.is_open()) {
      LOG(FATAL) << "failed to open log file for writing";
    }
    // write config
    std::ofstream configFile(getRunFile("config", runIdx, runPath));
    cereal::JSONOutputArchive ar(configFile);
    ar(CEREAL_NVP(config));
  }

  auto logStatus = [&logFile, isMaster](
                       TrainMeters& mtrs,
                       int64_t epoch,
                       int64_t nupdates,
                       double lr) {
    syncMeter(mtrs);

    if (isMaster) {
      auto logMsg = getLogString(
          mtrs, {}, epoch, nupdates, lr, 0 /* lrcrit */, 1 /* scaleFactor */);
      FL_LOG_MASTER(INFO) << logMsg;
      appendToLog(logFile, logMsg);
    }
  };

  auto saveModels = [&](int iter, int totalUpdates) {
    if (isMaster) {
      // Save last epoch
      config[kEpoch] = std::to_string(iter);
      config[kUpdates] = std::to_string(totalUpdates);

      std::string filename;
      if (FLAGS_itersave) {
        filename =
            getRunFile(format("model_iter_%03d.bin", iter), runIdx, runPath);
        Serializer::save(
            filename, FL_APP_ASR_VERSION, config, network, criterion, netoptim);
      }

      // save last model
      filename = getRunFile("model_last.bin", runIdx, runPath);
      Serializer::save(
          filename, FL_APP_ASR_VERSION, config, network, criterion, netoptim);

      // save if better than ever for one valid
      for (const auto& v : validminerrs) {
        double verr = meters.valid[v.first].wrdEdit.errorRate()[0];
        if (verr < validminerrs[v.first]) {
          validminerrs[v.first] = verr;
          std::string cleaned_v = cleanFilepath(v.first);
          std::string vfname =
              getRunFile("model_" + cleaned_v + ".bin", runIdx, runPath);
          Serializer::save(
              vfname, FL_APP_ASR_VERSION, config, network, criterion, netoptim);
        }
      }

      // print brief stats on memory allocation (so far)
      auto* curMemMgr =
          fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
      if (curMemMgr) {
        curMemMgr->printInfo("Memory Manager Stats", 0 /* device id */);
      }
    }
  };

  /* ===================== Hooks ===================== */
  auto evalOutput = [&tokenDict, &criterion](
                        const af::array& op,
                        const af::array& target,
                        DatasetMeters& mtr) {
    auto batchsz = op.dims(2);
    for (int b = 0; b < batchsz; ++b) {
      auto tgt = target(af::span, b);
      auto viterbipath =
          afToVector<int>(criterion->viterbiPath(op(af::span, af::span, b)));
      auto tgtraw = afToVector<int>(tgt);

      // Remove `-1`s appended to the target for batching (if any)
      auto labellen = getTargetSize(tgtraw.data(), tgtraw.size());
      tgtraw.resize(labellen);

      // remap actual, predicted targets for evaluating edit distance error

      auto ltrPred = tknPrediction2Ltr(
          viterbipath,
          tokenDict,
          FLAGS_criterion,
          FLAGS_surround,
          false, /* isSeq2seqCrit */
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);
      auto ltrTgt = tknTarget2Ltr(
          tgtraw,
          tokenDict,
          FLAGS_criterion,
          FLAGS_surround,
          false, /* isSeq2seqCrit */
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);

      auto wrdPred = tkn2Wrd(ltrPred, FLAGS_wordseparator);
      auto wrdTgt = tkn2Wrd(ltrTgt, FLAGS_wordseparator);

      mtr.tknEdit.add(ltrPred, ltrTgt);
      mtr.wrdEdit.add(wrdPred, wrdTgt);
    }
  };

  auto test = [&evalOutput](
                  std::shared_ptr<fl::Module> ntwrk,
                  std::shared_ptr<SequenceCriterion> crit,
                  std::shared_ptr<fl::Dataset> validds,
                  DatasetMeters& mtrs) {
    ntwrk->eval();
    mtrs.tknEdit.reset();
    mtrs.wrdEdit.reset();
    mtrs.loss.reset();

    auto curValidset = loadPrefetchDataset(
        validds, FLAGS_nthread, false /* shuffle */, 0 /* seed */);

    for (auto& batch : *curValidset) {
      auto output = fl::ext::forwardSequentialModuleWithPadMask(
          fl::input(batch[kInputIdx]), ntwrk, batch[kDurationIdx]);
      auto loss =
          crit->forward({output, fl::Variable(batch[kTargetIdx], false)})
              .front();
      mtrs.loss.add(loss.array());
      evalOutput(output.array(), batch[kTargetIdx], mtrs);
    }
  };

  int64_t curEpoch = startEpoch;

  auto train = [&meters,
                &test,
                &logStatus,
                &saveModels,
                &evalOutput,
                &validds,
                &curEpoch,
                &startUpdate,
                reducer](
                   std::shared_ptr<fl::Module> ntwrk,
                   std::shared_ptr<SequenceCriterion> crit,
                   std::shared_ptr<fl::Dataset> trainset,
                   std::shared_ptr<fl::FirstOrderOptimizer> netopt,
                   double initlr,
                   int64_t nbatches) {
    if (reducer) {
      fl::distributeModuleGrads(ntwrk, reducer);
    }

    meters.train.loss.reset();
    meters.train.tknEdit.reset();
    meters.train.wrdEdit.reset();

    std::shared_ptr<fl::Module> saug;
    if (FLAGS_saug_start_update >= 0) {
      saug = std::make_shared<fl::SpecAugment>(
          FLAGS_filterbanks,
          FLAGS_saug_fmaskf,
          FLAGS_saug_fmaskn,
          FLAGS_saug_tmaskt,
          FLAGS_saug_tmaskp,
          FLAGS_saug_tmaskn);
    }

    fl::allReduceParameters(ntwrk);

    auto resetTimeStatMeters = [&meters]() {
      meters.runtime.reset();
      meters.stats.reset();
      meters.sampletimer.reset();
      meters.fwdtimer.reset();
      meters.critfwdtimer.reset();
      meters.bwdtimer.reset();
      meters.optimtimer.reset();
      meters.timer.reset();
    };
    auto runValAndSaveModel =
        [&](int64_t totalEpochs, int64_t totalUpdates, double lr) {
          meters.runtime.stop();
          meters.timer.stop();
          meters.sampletimer.stop();
          meters.fwdtimer.stop();
          meters.critfwdtimer.stop();
          meters.bwdtimer.stop();
          meters.optimtimer.stop();

          // valid
          for (auto& vds : validds) {
            test(ntwrk, crit, vds.second, meters.valid[vds.first]);
          }

          // print status
          try {
            logStatus(meters, totalEpochs, totalUpdates, lr);
          } catch (const std::exception& ex) {
            LOG(ERROR) << "Error while writing logs: " << ex.what();
          }
          // save last and best models
          try {
            saveModels(totalEpochs, totalUpdates);
          } catch (const std::exception& ex) {
            LOG(FATAL) << "Error while saving models: " << ex.what();
          }
          // reset meters for next readings
          meters.train.loss.reset();
          meters.train.tknEdit.reset();
          meters.train.wrdEdit.reset();
        };

    int64_t curBatch = startUpdate;
    double scaleFactor =
        FLAGS_fl_amp_use_mixed_precision ? FLAGS_fl_amp_scale_factor : 1.;
    unsigned int kScaleFactorUpdateInterval =
        FLAGS_fl_amp_scale_factor_update_interval;
    unsigned int kMaxScaleFactor = FLAGS_fl_amp_max_scale_factor;
    unsigned short scaleCounter = 1;
    while (curBatch < nbatches) {
      ++curEpoch; // counts partial epochs too!
      int64_t epochsAfterDecay = curEpoch - FLAGS_lr_decay;
      double lrDecayScale = std::pow(
          0.5,
          (epochsAfterDecay < 0 ? 0
                                : 1 + epochsAfterDecay / FLAGS_lr_decay_step));
      ntwrk->train();
      if (FLAGS_reportiters == 0) {
        resetTimeStatMeters();
      }
      std::hash<std::string> hasher;
      FL_LOG_MASTER(INFO) << "Shuffling trainset";
      auto curTrainset = loadPrefetchDataset(
          trainset, FLAGS_nthread, true /* shuffle */, curEpoch /* seed */);
      fl::sync();
      meters.sampletimer.resume();
      meters.runtime.resume();
      meters.timer.resume();
      FL_LOG_MASTER(INFO) << "Epoch " << curEpoch << " started!";
      for (auto& batch : *curTrainset) {
        ++curBatch;
        double lrScheduleScale;
        if (FLAGS_lrcosine) {
          const double pi = std::acos(-1);
          lrScheduleScale =
              std::cos(((double)curBatch) / ((double)nbatches) * pi / 2.0);
        } else {
          lrScheduleScale =
              std::pow(FLAGS_gamma, (double)curBatch / (double)FLAGS_stepsize);
        }
        netopt->setLr(
            initlr * lrDecayScale * lrScheduleScale *
            std::min(curBatch / double(FLAGS_warmup), 1.0));
        fl::sync();
        meters.timer.incUnit();
        meters.sampletimer.stopAndIncUnit();
        meters.stats.add(batch[kDurationIdx], batch[kTargetSizeIdx]);
        if (af::anyTrue<bool>(af::isNaN(batch[kInputIdx])) ||
            af::anyTrue<bool>(af::isNaN(batch[kTargetIdx]))) {
          LOG(FATAL) << "Sample has NaN values - "
                     << join(",", readSampleIds(batch[kSampleIdx]));
        }

        // Ensure no samples are skipped while adjusting the loss scale factor.
        // When gradient values are Inf/NaN, the model update is skipped and the
        // scale factor is adjusted accordingly for determinism.
        // The AMP algorithm implemented here mirrors:
        // - https://arxiv.org/abs/1710.03740
        // - https://bit.ly/35F5GqX
        // - https://bit.ly/3mn2qr0
        bool retrySample = false;
        do {
          retrySample = false;
          // forward
          meters.fwdtimer.resume();
          auto input = fl::input(batch[kInputIdx]);
          if (FLAGS_saug_start_update >= 0 &&
              curBatch >= FLAGS_saug_start_update) {
            input = saug->forward({input}).front();
          }
          auto output = fl::ext::forwardSequentialModuleWithPadMask(
              input, ntwrk, batch[kDurationIdx]);
          fl::sync();
          meters.critfwdtimer.resume();
          auto loss =
              crit->forward({output, fl::noGrad(batch[kTargetIdx])}).front();
          fl::sync();
          meters.fwdtimer.stopAndIncUnit();
          meters.critfwdtimer.stopAndIncUnit();

          if (FLAGS_fl_amp_use_mixed_precision) {
            ++scaleCounter;
            loss = loss * scaleFactor;
          }

          if (af::anyTrue<bool>(af::isNaN(loss.array())) ||
              af::anyTrue<bool>(af::isInf(loss.array()))) {
            if (FLAGS_fl_amp_use_mixed_precision &&
                scaleFactor >= fl::kAmpMinimumScaleFactorValue) {
              scaleFactor = scaleFactor / 2.0f;
              FL_VLOG(2) << "AMP: Scale factor decreased. New value:\t"
                         << scaleFactor;
              scaleCounter = 1;
              retrySample = true;
              continue;
            } else {
              LOG(FATAL) << "Loss has NaN values. Samples - "
                         << join(",", readSampleIds(batch[kSampleIdx]));
            }
          }

          if (hasher(join(",", readSampleIds(batch[kSampleIdx]))) % 100 <=
              FLAGS_pcttraineval) {
            evalOutput(output.array(), batch[kTargetIdx], meters.train);
          }

          // backward
          meters.bwdtimer.resume();
          netopt->zeroGrad();
          loss.backward();
          if (reducer) {
            reducer->finalize();
          }
          fl::sync();
          meters.bwdtimer.stopAndIncUnit();

          // optimizer
          meters.optimtimer.resume();

          // scale down gradients by batchsize
          for (const auto& p : ntwrk->params()) {
            if (!p.isGradAvailable()) {
              continue;
            }
            p.grad() = p.grad() / (FLAGS_batchsize * scaleFactor);
            if (FLAGS_fl_amp_use_mixed_precision) {
              if (af::anyTrue<bool>(af::isNaN(p.grad().array())) ||
                  af::anyTrue<bool>(af::isInf(p.grad().array()))) {
                if (scaleFactor >= fl::kAmpMinimumScaleFactorValue) {
                  scaleFactor = scaleFactor / 2.0f;
                  FL_VLOG(2) << "AMP: Scale factor decreased. New value:\t"
                             << scaleFactor;
                  retrySample = true;
                }
                scaleCounter = 1;
                break;
              }
            }
          }
          if (retrySample) {
            meters.optimtimer.stop();
            continue;
          }

          meters.train.loss.add((loss / scaleFactor).array());

        } while (retrySample);

        // clamp gradients
        if (FLAGS_maxgradnorm > 0) {
          auto params = ntwrk->params();

          fl::clipGradNorm(params, FLAGS_maxgradnorm);
        }

        // update weights
        netopt->step();
        fl::sync();
        meters.optimtimer.stopAndIncUnit();

        // update scale factor
        if (FLAGS_fl_amp_use_mixed_precision && scaleFactor < kMaxScaleFactor) {
          if (scaleCounter % kScaleFactorUpdateInterval == 0) {
            scaleFactor *= 2;
            FL_VLOG(2) << "AMP: Scale factor doubled. New value:\t"
                       << scaleFactor;
          } else {
            scaleFactor += 2;
            FL_VLOG(3) << "AMP: Scale factor incremented. New value\t"
                       << scaleFactor;
          }
        }

        meters.sampletimer.resume();

        if (FLAGS_reportiters > 0 && curBatch % FLAGS_reportiters == 0) {
          runValAndSaveModel(curEpoch, curBatch, netopt->getLr());
          resetTimeStatMeters();
          ntwrk->train();
          meters.sampletimer.resume();
          meters.runtime.resume();
          meters.timer.resume();
        }
        if (curBatch > nbatches) {
          break;
        }
      }
      fl::sync();
      if (FLAGS_reportiters == 0) {
        runValAndSaveModel(curEpoch, curBatch, netopt->getLr());
      }
    }
  };

  /* ===================== Train ===================== */

  train(network, criterion, trainds, netoptim, FLAGS_lr, FLAGS_iter);

  FL_LOG_MASTER(INFO) << "Finished training";
  return 0;
}
