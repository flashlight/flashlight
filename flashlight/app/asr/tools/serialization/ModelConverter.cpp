/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>
#include <string.h>
#include <memory>
#include <unordered_map>

#include <flashlight/fl/flashlight.h>

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/common/Flags.h"
#include "flashlight/app/asr/criterion/criterion.h"
#include "flashlight/app/asr/data/FeatureTransforms.h"
#include "flashlight/app/asr/data/Utils.h"
#include "flashlight/app/asr/runtime/runtime.h"
#include "flashlight/app/asr/tools/serialization/Compat.h"
#include "flashlight/ext/common/SequentialBuilder.h"
#include "flashlight/ext/common/Serializer.h"
#include "flashlight/ext/plugin/ModulePlugin.h"

using namespace fl::app::asr;

namespace {

auto newModelPath = [](const std::string& path) { return path + ".new"; };

auto tempModelPath = [](const std::string& path) { return path + ".tmp"; };

void loadFromBinaryDump(
    const char* fname,
    std::shared_ptr<fl::Module> ntwrk,
    std::shared_ptr<fl::Module> crit) {
  ntwrk->eval();
  crit->eval();
  for (int i = 0; i < ntwrk->params().size(); ++i) {
    std::string key = "net-" + std::to_string(i);
    ntwrk->setParams(fl::Variable(af::readArray(fname, key.c_str()), false), i);
  }
  for (int i = 0; i < crit->params().size(); ++i) {
    std::string key = "crt-" + std::to_string(i);
    crit->setParams(fl::Variable(af::readArray(fname, key.c_str()), false), i);
  }
}

void saveToBinaryDump(
    const char* fname,
    std::shared_ptr<fl::Module> ntwrk,
    std::shared_ptr<fl::Module> crit) {
  ntwrk->eval();
  crit->eval();
  for (int i = 0; i < ntwrk->params().size(); ++i) {
    std::string key = "net-" + std::to_string(i);
    af::saveArray(key.c_str(), ntwrk->param(i).array(), fname, (i != 0));
  }

  for (int i = 0; i < crit->params().size(); ++i) {
    std::string key = "crt-" + std::to_string(i);
    af::saveArray(key.c_str(), crit->param(i).array(), fname, true);
  }
}

int getSpeechFeatureSize() {
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
  return featureRes.first;
}

} // namespace

int main(int argc, char** argv) {
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;

  initCompat();

  if (argc < 3) {
    LOG(FATAL)
        << "Incorrect usage. 'fl_asr_model_converter [model_path] [old/new]'";
  }

  std::string binaryType = argv[1];
  std::string modelPath = argv[2];
  std::string version;
  if (binaryType == "old") {
    LOG(INFO) << "Saving params from `old binary` model to a binary dump";
    fl::ext::Serializer::load(modelPath, version, cfg, network, criterion);
    saveToBinaryDump(tempModelPath(modelPath).c_str(), network, criterion);
  } else if (binaryType == "new") {
    LOG(INFO) << "Loading model params from binary dump to `new binary` model";

    // Read gflags from old model
    fl::ext::Serializer::load(modelPath, version, cfg);
    auto flags = cfg.find(kGflags);
    LOG_IF(FATAL, flags == cfg.end()) << "Invalid config loaded";
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    handleDeprecatedFlags();

    auto numFeatures = getSpeechFeatureSize();

    fl::lib::text::Dictionary tokenDict(FLAGS_tokens);
    auto scalemode = getCriterionScaleMode(FLAGS_onorm, FLAGS_sqnorm);
    // Setup-specific modifications
    for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
      tokenDict.addEntry(std::to_string(r));
    }
    // ctc expects the blank label last
    if (FLAGS_criterion == kCtcCriterion) {
      tokenDict.addEntry(kBlankToken);
    }
    bool isSeq2seqCrit = FLAGS_criterion == kSeq2SeqTransformerCriterion ||
        FLAGS_criterion == kSeq2SeqRNNCriterion;
    if (isSeq2seqCrit) {
      tokenDict.addEntry(fl::app::asr::kEosToken);
      tokenDict.addEntry(fl::lib::text::kPadToken);
    }

    int numClasses = tokenDict.indexSize();

    // Intialize Network and Criterion
    if (fl::lib::endsWith(FLAGS_arch, ".so")) {
      network = fl::ext::ModulePlugin(FLAGS_arch).arch(numFeatures, numClasses);
    } else {
      network =
          fl::ext::buildSequentialModule(FLAGS_arch, numFeatures, numClasses);
    }
    if (FLAGS_criterion == kCtcCriterion) {
      criterion = std::make_shared<CTCLoss>(scalemode);
    } else if (FLAGS_criterion == kAsgCriterion) {
      criterion =
          std::make_shared<ASGLoss>(numClasses, scalemode, FLAGS_transdiag);
    } else if (FLAGS_criterion == kSeq2SeqRNNCriterion) {
      std::vector<std::shared_ptr<AttentionBase>> attentions;
      for (int i = 0; i < FLAGS_decoderattnround; i++) {
        attentions.push_back(createAttention());
      }
      criterion = std::make_shared<Seq2SeqCriterion>(
          numClasses,
          FLAGS_encoderdim,
          tokenDict.getIndex(fl::app::asr::kEosToken),
          tokenDict.getIndex(fl::lib::text::kPadToken),
          FLAGS_maxdecoderoutputlen,
          attentions,
          createAttentionWindow(),
          FLAGS_trainWithWindow,
          FLAGS_pctteacherforcing,
          FLAGS_labelsmooth,
          FLAGS_inputfeeding,
          FLAGS_samplingstrategy,
          FLAGS_gumbeltemperature,
          FLAGS_decoderrnnlayer,
          FLAGS_decoderattnround,
          FLAGS_decoderdropout);
    } else if (FLAGS_criterion == kSeq2SeqTransformerCriterion) {
      criterion = std::make_shared<TransformerCriterion>(
          numClasses,
          FLAGS_encoderdim,
          tokenDict.getIndex(fl::app::asr::kEosToken),
          tokenDict.getIndex(fl::lib::text::kPadToken),
          FLAGS_maxdecoderoutputlen,
          FLAGS_am_decoder_tr_layers,
          createAttention(),
          createAttentionWindow(),
          FLAGS_trainWithWindow,
          FLAGS_labelsmooth,
          FLAGS_pctteacherforcing,
          FLAGS_am_decoder_tr_dropout,
          FLAGS_am_decoder_tr_layerdrop);
    } else {
      LOG(FATAL) << "unimplemented criterion";
    }

    loadFromBinaryDump(tempModelPath(modelPath).c_str(), network, criterion);
    fl::ext::Serializer::save(
        newModelPath(modelPath), FL_APP_ASR_VERSION, cfg, network, criterion);

  } else {
    LOG(FATAL) << "Incorrect binary type specified.";
  }

  LOG(INFO) << "Done !";

  return 0;
}
