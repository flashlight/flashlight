/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/criterion/criterion.h"
#include "flashlight/pkg/speech/data/FeatureTransforms.h"
#include "flashlight/pkg/speech/data/Utils.h"
#include "flashlight/pkg/speech/runtime/runtime.h"
#include "flashlight/app/asr/tools/alignment/Utils.h"
#include "flashlight/fl/common/SequentialBuilder.h"
#include "flashlight/fl/common/Serializer.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/dictionary/Defines.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"

using namespace fl::app::asr;
using namespace fl::lib;
using namespace fl::app::asr::alignment;

int main(int argc, char** argv) {
  std::string exec(argv[0]);
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage("Usage: \n " + exec + " align_file_path [flags]");
  if (argc <= 2) {
    LOG(FATAL) << gflags::ProgramUsage();
  }
  std::string alignFilePath = argv[1];

  fl::init();

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  std::string version;
  LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
  fl::ext::Serializer::load(FLAGS_am, version, cfg, network, criterion);
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
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  text::Dictionary tokenDict(FLAGS_tokens);
  // Setup-specific modifications
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry("<" + std::to_string(r) + ">");
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
  LOG(INFO) << "Number of classes (network): " << numClasses;

  text::DictionaryMap dicts;
  dicts.insert({kTargetIdx, tokenDict});

  std::mutex write_mutex;
  std::ofstream alignFile;
  alignFile.open(alignFilePath);
  if (!alignFile.is_open() || !alignFile.good()) {
    LOG(FATAL) << "Error opening log file";
  } else {
    LOG(INFO) << "Writing alignment to: " << alignFilePath;
  }

  fl::lib::text::Dictionary wordDict;
  text::LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = text::loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = text::createWordDict(lexicon);
    LOG(INFO) << "Number of words: " << wordDict.indexSize();
    wordDict.setDefaultIndex(wordDict.getIndex(text::kUnkToken));
  }

  LOG(INFO) << "Loaded lexicon";

  auto writeLog = [&](const std::string& logStr) {
    std::lock_guard<std::mutex> lock(write_mutex);
    alignFile << logStr;
    if (FLAGS_show) {
      std::cout << logStr;
    }
  };

  /* ===================== Create Dataset ===================== */
  int worldRank = 0;
  int worldSize = 1;
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
      isSeq2seqCrit,
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
  int targetpadVal = isSeq2seqCrit
      ? tokenDict.getIndex(fl::lib::text::kPadToken)
      : kTargetPadValue;
  int wordpadVal = kTargetPadValue;

  auto ds = createDataset(
      {FLAGS_test},
      FLAGS_datadir,
      1,
      inputTransform,
      targetTransform,
      wordTransform,
      std::make_tuple(0, targetpadVal, wordpadVal),
      worldRank,
      worldSize);

  LOG(INFO) << "[Dataset] Dataset loaded";

  auto postprocessFN = getWordSegmenter(criterion);

  int batches = 0;
  fl::TimeMeter alignMtr;
  fl::TimeMeter fwdMtr;
  fl::TimeMeter parseMtr;

  for (auto& sample : *ds) {
    fwdMtr.resume();
    const auto input = fl::input(sample[kInputIdx]);
    fl::Variable rawEmission = fl::ext::forwardSequentialModuleWithPadMask(
        input, network, sample[kDurationIdx]);
    fwdMtr.stop();
    alignMtr.resume();
    auto bestPaths = criterion->viterbiPathWithTarget(
        rawEmission.array(), sample[kTargetIdx]);
    alignMtr.stop();
    parseMtr.resume();

    const double timeScale =
        static_cast<double>(input.dims(0)) / rawEmission.dims(1);

    const std::vector<std::vector<std::string>> tokenPaths =
        mapIndexToToken(bestPaths, dicts);
    const std::vector<std::string> sampleIdsStr =
        readSampleIds(sample[kSampleIdx]);

    for (int b = 0; b < tokenPaths.size(); b++) {
      if (sampleIdsStr.size() > b) {
        const std::vector<std::string>& path = tokenPaths[b];
        const std::vector<AlignedWord> segmentation = postprocessFN(
            path, FLAGS_replabel, FLAGS_framestridems * timeScale);
        const std::string ctmString = getCTMFormat(segmentation);
        std::stringstream buffer;
        buffer << sampleIdsStr[b] << "\t" << ctmString << "\n";
        writeLog(buffer.str());
      }
    }
    parseMtr.stop();
    ++batches;
    if (batches % 500 == 0) {
      LOG(INFO) << "Done samples: " << batches;
    }
  }

  LOG(INFO) << "Align time: " << alignMtr.value();
  LOG(INFO) << "Fwd time: " << fwdMtr.value();
  LOG(INFO) << "Parse time: " << parseMtr.value();
  alignFile.close();
  return 0;
}
