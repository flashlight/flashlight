/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Performs voice-activity detection with a CTC model.
 * For each input sample in the dataset, outputs the following:
 * - Chunk level probabilities of non-speech based on the probability of a
 *   blank label assigned as per the acoustic model trained with CTC. These are
 *   assigned for each chunk of output. For stride 1 model, these will be each
 *   frame (10 ms), but for a model with stride 8, these will be (80 ms)
 *   intervals (output in .vad file for each sample)
 * - The perplexity of the predicted sequence based on a specified input
 *   language model (first output in .sts file for each sample)
 * - The percentage of the audio containing speech based on the passed
 *   --vad_threshold flag (second output in .sts file for each sample)
 * - The most likely token-level transcription of given audio based on the
 *   acoustic model output only (output in .tsc file for each sample).
 * - Frame wise token emissions based on the most-likely token emitted for each
 *   chunk, (output in .fwt file for each sample).
 */

#include <stdlib.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/criterion/criterion.h"
#include "flashlight/pkg/speech/data/FeatureTransforms.h"
#include "flashlight/pkg/speech/data/Utils.h"
#include "flashlight/pkg/speech/decoder/TranscriptionUtils.h"
#include "flashlight/pkg/speech/runtime/runtime.h"
#include "flashlight/ext/common/SequentialBuilder.h"
#include "flashlight/ext/common/Serializer.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"

namespace {

DEFINE_double(
    vad_threshold,
    0.99,
    "Blank probability threshold at which a frame is deemed voice-inactive");
DEFINE_string(outpath, "", "Output path for generated results files");

// Extensions for each output file
const std::string kVadExt = ".vad";
const std::string kTknFrameWiseTokensExt = ".fwt";
const std::string kLtrTranscriptExt = ".tsc";
const std::string kPerplexityPctSpeechExt = ".sts";

} // namespace

using namespace fl::app::asr;
using namespace fl::lib;
using namespace fl::ext;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage("Usage: Please refer to https://git.io/JLbJ6");
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  fl::init();

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  auto flagsfile = FLAGS_flagsfile;
  if (!flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << flagsfile;
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  std::string version;
  LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
  fl::ext::Serializer::load(FLAGS_am, version, cfg, network, criterion);
  if (version != FL_APP_ASR_VERSION) {
    LOG(WARNING) << "[Network] Model version " << version
                 << " and code version " << FL_APP_ASR_VERSION;
  }
  network->eval();
  criterion->eval();

  LOG(INFO) << "[Network] " << network->prettyString();
  LOG(INFO) << "[Criterion] " << criterion->prettyString();
  LOG(INFO) << "[Network] Number of params: " << numTotalParams(network);

  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "[Network] Invalid config loaded from " << FLAGS_am;
  }
  LOG(INFO) << "[Network] Updating flags from config file: " << FLAGS_am;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

  // override with user-specified flags
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!flagsfile.empty()) {
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  fl::lib::text::Dictionary tokenDict(FLAGS_tokens);
  // Setup-specific modifications
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry("<" + std::to_string(r) + ">");
  }
  // ctc expects the blank label last
  if (FLAGS_criterion == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  } else {
    LOG(FATAL) << "CTC-trained model required for VAD-CTC.";
  }

  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  fl::lib::text::Dictionary wordDict;
  text::LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = text::loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = text::createWordDict(lexicon);
    LOG(INFO) << "Number of words: " << wordDict.indexSize();
    wordDict.setDefaultIndex(wordDict.getIndex(text::kUnkToken));
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
  FeatureType featType =
      getFeatureType(FLAGS_features_type, FLAGS_channels, featParams).second;

  TargetGenerationConfig targetGenConfig(
      FLAGS_wordseparator,
      FLAGS_sampletarget,
      FLAGS_criterion,
      FLAGS_surround,
      false /* isSeq2SeqCrit */,
      FLAGS_replabel,
      true /* skip unk */,
      FLAGS_usewordpiece /* fallback2LetterWordSepLeft */,
      !FLAGS_usewordpiece /* fallback2LetterWordSepLeft */);

  auto inputTransform = inputFeatures(
      featParams,
      featType,
      {FLAGS_localnrmlleftctx, FLAGS_localnrmlrightctx},
      {});
  auto targetTransform = targetFeatures(tokenDict, lexicon, targetGenConfig);
  auto wordTransform = wordFeatures(wordDict);
  int targetpadVal = kTargetPadValue;
  int wordpadVal = kTargetPadValue;

  auto ds = createDataset(
      {FLAGS_test},
      FLAGS_datadir,
      1,
      inputTransform,
      targetTransform,
      wordTransform,
      std::make_tuple(0, targetpadVal, wordpadVal),
      0,
      1);
  LOG(INFO) << "[Dataset] Dataset loaded.";

  /* ===================== Build LM ===================== */
  std::shared_ptr<text::LM> lm;
  if (!FLAGS_lm.empty()) {
    if (FLAGS_lmtype == "kenlm") {
      lm = std::make_shared<text::KenLM>(FLAGS_lm, wordDict);
      if (!lm) {
        throw std::runtime_error(
            "[LM constructing] Failed to load LM: " + FLAGS_lm);
      }
    } else {
      throw std::runtime_error(
          "[LM constructing] Invalid LM Type: " + FLAGS_lmtype);
    }
  }

  /* ===================== Test ===================== */
  int cnt = 0;
  auto prefetchds =
      loadPrefetchDataset(ds, FLAGS_nthread, false /* shuffle */, 0 /* seed */);
  for (auto& sample : *prefetchds) {
    auto rawEmission = fl::ext::forwardSequentialModuleWithPadMask(
        fl::input(sample[kInputIdx]), network, sample[kDurationIdx]);
    auto sampleId = readSampleIds(sample[kSampleIdx]).front();
    LOG(INFO) << "Processing sample ID " << sampleId;

    // Hypothesis
    auto tokenPrediction =
        afToVector<int>(criterion->viterbiPath(rawEmission.array()));
    auto letterPrediction = tknPrediction2Ltr(
        tokenPrediction,
        tokenDict,
        FLAGS_criterion,
        FLAGS_surround,
        false /* isSeq2SeqCrit */,
        FLAGS_replabel,
        FLAGS_usewordpiece,
        FLAGS_wordseparator);
    std::vector<std::string> wordPrediction =
        tkn2Wrd(letterPrediction, FLAGS_wordseparator);

    float lmScore = 0;
    if (!FLAGS_lm.empty()) {
      // LM score
      auto inState = lm->start(0);
      for (const auto& word : wordPrediction) {
        auto lmReturn = lm->score(inState, wordDict.getIndex(word));
        inState = lmReturn.first;
        lmScore += lmReturn.second;
      }
      auto lmReturn = lm->finish(inState);
      lmScore += lmReturn.second;
    }

    // Determine results basename. In case the sample id contains an extension,
    // else a noop
    fl::lib::dirCreateRecursive(FLAGS_outpath);
    auto baseName = fl::lib::pathsConcat(
        FLAGS_outpath, sampleId.substr(0, sampleId.find_last_of(".")));

    // Output chunk-level tokens outputs (or blanks)
    std::ofstream tknOutStream(baseName + kTknFrameWiseTokensExt);
    for (auto token : tokenPrediction) {
      tknOutStream << tokenDict.getEntry(token) << " ";
    }
    tknOutStream << std::endl;
    tknOutStream.close();

    int blank = tokenDict.getIndex(kBlankToken);
    int N = rawEmission.dims(0);
    int T = rawEmission.dims(1);
    float vadFrameCnt = 0;
    auto emissions = afToVector<float>(softmax(rawEmission, 0).array());
    for (int i = 0; i < T; i++) {
      if (emissions[i * N + blank] < FLAGS_vad_threshold) {
        vadFrameCnt += 1;
      }
    }

    // Output chunk-level VAD probabilities
    std::ofstream vadProbOutStream(baseName + kVadExt);
    for (int i = 0; i < T; i++) {
      vadProbOutStream << std::setprecision(4) << emissions[i * N + blank]
                       << " ";
    }
    vadProbOutStream << std::endl;
    vadProbOutStream.close();

    // Token transcript
    std::ofstream tScriptOutStream(baseName + kLtrTranscriptExt);
    tScriptOutStream << join("", letterPrediction) << std::endl;
    tScriptOutStream.close();

    // Perplexity under the given LM and % of audio containing speech given VAD
    // threshold
    std::ofstream statsOutStream(baseName + kPerplexityPctSpeechExt);
    statsOutStream << sampleId << " " << vadFrameCnt / T;
    if (!FLAGS_lm.empty()) {
      statsOutStream << " " << std::pow(10.0, -lmScore / wordPrediction.size());
    }
    statsOutStream << std::endl;
    statsOutStream.close();

    ++cnt;
    if (cnt == FLAGS_maxload) {
      break;
    }
  }

  return 0;
}
