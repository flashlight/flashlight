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
 *   --vadthreshold flag (second output in .sts file for each sample)
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

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/criterion/criterion.h"
#include "flashlight/app/asr/decoder/TranscriptionUtils.h"
#include "flashlight/app/asr/runtime/runtime.h"
#include "flashlight/ext/common/Serializer.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"

namespace {

DEFINE_double(
    vadthreshold,
    0.99,
    "Blank probability threshold at which a frame is deemed voice-inactive");
DEFINE_string(outpath, "", "Output path for generated results files");

// Extensions for each output file
const std::string kVadExt = ".vad";
const std::string kFrameWiseTokensExt = ".fwt";
const std::string kWordPieceTranscriptExt = ".tsc";
const std::string kPerplexityPctSpeechExt = ".sts";

} // namespace

using namespace fl::app::asr;
using namespace fl::lib;
using namespace fl::ext;

int main(int argc, char** argv) {
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage("Usage: Please refer to https://git.io/Je9lG");
  if (argc <= 1) {
    FL_LOG(fl::FATAL) << gflags::ProgramUsage();
  }

  /* ===================== Parse Options ===================== */
  FL_LOG(fl::INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  auto flagsfile = FLAGS_flagsfile;
  if (!flagsfile.empty()) {
    FL_LOG(fl::INFO) << "Reading flags from file " << flagsfile;
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  std::string version;
  FL_LOG(fl::INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
  fl::ext::Serializer::load(FLAGS_am, version, cfg, network, criterion);
  if (version != FL_APP_ASR_VERSION) {
    FL_LOG(fl::WARNING) << "[Network] Model version " << version
                        << " and code version " << FL_APP_ASR_VERSION;
  }
  network->eval();
  criterion->eval();

  FL_LOG(fl::INFO) << "[Network] " << network->prettyString();
  FL_LOG(fl::INFO) << "[Criterion] " << criterion->prettyString();
  FL_LOG(fl::INFO) << "[Network] Number of params: " << numTotalParams(network);

  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    FL_LOG(fl::FATAL) << "[Network] Invalid config loaded from " << FLAGS_am;
  }
  FL_LOG(fl::INFO) << "[Network] Updating flags from config file: " << FLAGS_am;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

  // override with user-specified flags
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!flagsfile.empty()) {
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }

  FL_LOG(fl::INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  auto dictPath = fl::lib::pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error(
        "Invalid dictionary filepath specified " + dictPath);
  }
  fl::lib::text::Dictionary tokenDict(dictPath);
  // Setup-specific modifications
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry("<" + std::to_string(r) + ">");
  }
  // ctc expects the blank label last
  if (FLAGS_criterion == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  } else {
    FL_LOG(fl::FATAL) << "CTC-trained model required for VAD-CTC.";
  }
  if (FLAGS_eostoken) {
    tokenDict.addEntry(kEosToken);
  }

  int numClasses = tokenDict.indexSize();
  FL_LOG(fl::INFO) << "Number of classes (network): " << numClasses;

  fl::lib::text::Dictionary wordDict;
  text::LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = text::loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = text::createWordDict(lexicon);
    FL_LOG(fl::INFO) << "Number of words: " << wordDict.indexSize();
    wordDict.setDefaultIndex(wordDict.getIndex(text::kUnkToken));
  }

  text::DictionaryMap dicts = {{kTargetIdx, tokenDict}, {kWordIdx, wordDict}};

  /* ===================== Create Dataset ===================== */
  auto ds = createDataset(FLAGS_test, dicts, lexicon, 1, 0, 1);
  FL_LOG(fl::INFO) << "[Dataset] Dataset loaded.";

  /* ===================== Build LM ===================== */
  std::shared_ptr<text::LM> lm;
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

  /* ===================== Test ===================== */
  int cnt = 0;
  for (auto& sample : *ds) {
    auto rawEmission = network->forward({fl::input(sample[kInputIdx])}).front();
    auto sampleId = readSampleIds(sample[kSampleIdx]).front();
    FL_LOG(fl::INFO) << "Processing sample ID " << sampleId;

    // Hypothesis
    auto tokenPrediction =
        afToVector<int>(criterion->viterbiPath(rawEmission.array()));
    auto letterPrediction = tknPrediction2Ltr(
        tokenPrediction,
        tokenDict,
        FLAGS_criterion,
        FLAGS_surround,
        FLAGS_eostoken,
        FLAGS_replabel,
        FLAGS_usewordpiece,
        FLAGS_wordseparator);
    std::vector<std::string> wordPrediction =
        tkn2Wrd(letterPrediction, FLAGS_wordseparator);

    // LM score
    float lmScore = 0;
    auto inState = lm->start(0);
    for (const auto& word : wordPrediction) {
      auto lmReturn = lm->score(inState, wordDict.getIndex(word));
      inState = lmReturn.first;
      lmScore += lmReturn.second;
    }
    auto lmReturn = lm->finish(inState);
    lmScore += lmReturn.second;

    // Determine results basename. In case the sample id contains an extension,
    // else a noop
    auto baseName = fl::lib::pathsConcat(
        FLAGS_outpath, sampleId.substr(0, sampleId.find_last_of(".")));

    // Output chunk-level word piece outputs (or blanks)
    std::ofstream wpOutStream(baseName + kFrameWiseTokensExt);
    for (auto token : tokenPrediction) {
      wpOutStream << tokenDict.getEntry(token) << " ";
    }
    wpOutStream << std::endl;
    wpOutStream.close();

    int blank = tokenDict.getIndex(kBlankToken);
    int N = rawEmission.dims(0);
    int T = rawEmission.dims(1);
    float vadFrameCnt = 0;
    auto emissions = afToVector<float>(softmax(rawEmission, 0).array());
    for (int i = 0; i < T; i++) {
      if (emissions[i * N + blank] < FLAGS_vadthreshold) {
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

    // Word-piece transcript
    std::ofstream tScriptOutStream(baseName + kWordPieceTranscriptExt);
    tScriptOutStream << join("", letterPrediction) << std::endl;
    tScriptOutStream.close();

    // Perplexity under the given LM and % of audio containing speech given VAD
    // threshold
    std::ofstream statsOutStream(baseName + kPerplexityPctSpeechExt);
    statsOutStream << sampleId << " "
                   << std::pow(10.0, -lmScore / wordPrediction.size()) << " "
                   << vadFrameCnt / T << std::endl;
    statsOutStream.close();

    ++cnt;
    if (cnt == FLAGS_maxload) {
      break;
    }
  }

  return 0;
}
