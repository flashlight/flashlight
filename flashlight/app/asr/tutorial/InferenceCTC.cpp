/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>
#include <unordered_map>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/fl/flashlight.h"

#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight/pkg/runtime/common/DistributedUtils.h"
#include "flashlight/pkg/runtime/common/SequentialBuilder.h"
#include "flashlight/pkg/runtime/common/Serializer.h"
#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/data/FeatureTransforms.h"
#include "flashlight/pkg/speech/data/Sound.h"
#include "flashlight/pkg/speech/data/Utils.h"
#include "flashlight/pkg/speech/decoder/DecodeUtils.h"
#include "flashlight/pkg/speech/decoder/Defines.h"
#include "flashlight/pkg/speech/decoder/TranscriptionUtils.h"

DEFINE_string(
    am_path,
    "",
    "Path to CTC trained acousitc mode to perform inference");
DEFINE_string(tokens_path, "", "Path to the model tokens set");
DEFINE_string(
    lexicon_path,
    "",
    "Path to the lexicon which defines mapping between word and tokens + restricts beam search");
DEFINE_string(
    lm_path,
    "",
    "Path to ngram language model. Either arpa file or KenLM bin file");
DEFINE_int32(beam_size, 100, "Beam size for the beam-search decoding");
DEFINE_int32(
    beam_size_token,
    10,
    "Tokens beam size for the beam-search decoding");
DEFINE_double(beam_threshold, 100, "Beam-search decoding pruning parameters");
DEFINE_double(lm_weight, 3, "Beam-search decoding language model weight");
DEFINE_double(word_score, 0, "Beam-search decoding word addition score");
DEFINE_int64(sample_rate, 16000, "Sample rate of the input audio");
DEFINE_string(
    audio_list,
    "",
    "Path to the file where each row is audio file path, "
    "if it is empty interactive regime is used");

void serializeAndCheckFlags() {
  std::stringstream serialized;
  std::vector<gflags::CommandLineFlagInfo> allFlags;
  std::string currVal;
  gflags::GetAllFlags(&allFlags);
  for (auto itr = allFlags.begin(); itr != allFlags.end(); ++itr) {
    gflags::GetCommandLineOption(itr->name.c_str(), &currVal);
    serialized << "--" << itr->name << "=" << currVal << ";";
  }
  LOG(INFO) << "Gflags after parsing\n" << serialized.str();

  std::unordered_map<std::string, std::string> flgs = {
      {"am_path", FLAGS_am_path},
      {"tokens_path", FLAGS_tokens_path},
      {"lexicon_path", FLAGS_lexicon_path},
      {"lm_path", FLAGS_lm_path}};
  for (auto& path : flgs) {
    if (path.second.empty() || !fs::exists(path.second)) {
      throw std::runtime_error(
          "[Inference tutorial for CTC] Invalid file path specified for the flag --" +
          path.first + " with value '" + path.second +
          "': either it is empty or doesn't exist.");
    }
  }
}

void loadModel(
    std::shared_ptr<fl::Module>& network,
    std::unordered_map<std::string, std::string>& networkFlags) {
  std::unordered_map<std::string, std::string> cfg;
  std::string version;

  LOG(INFO) << "[Inference tutorial for CTC] Reading acoustic model from "
            << FLAGS_am_path;
  fl::setDevice(0);
  fl::pkg::runtime::Serializer::load(FLAGS_am_path, version, cfg, network);
  if (version != FL_APP_ASR_VERSION) {
    LOG(WARNING) << "[Inference tutorial for CTC] Acostuc model version "
                 << version << " and code version " << FL_APP_ASR_VERSION;
  }
  if (cfg.find(fl::pkg::speech::kGflags) == cfg.end()) {
    LOG(FATAL)
        << "[Inference tutorial for CTC] Invalid config is loaded from acoustic model"
        << FLAGS_am_path;
  }
  for (auto line : fl::lib::split("\n", cfg[fl::pkg::speech::kGflags])) {
    if (line == "") {
      continue;
    }
    auto res = fl::lib::split("=", line);
    if (res.size() >= 2) {
      auto key = fl::lib::split("--", res[0])[1];
      networkFlags[key] = res[1];
    }
  }
  if (networkFlags["criterion"] != fl::pkg::speech::kCtcCriterion) {
    LOG(FATAL)
        << "[Inference tutorial for CTC]: provided model is trained not with CTC, but with "
        << networkFlags["criterion"]
        << ". This type is not supported in the tutorial";
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }

  fl::init();

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  serializeAndCheckFlags();
  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::unordered_map<std::string, std::string> networkFlags;
  loadModel(network, networkFlags);
  network->eval();
  LOG(INFO) << "[Inference tutorial for CTC] Network is loaded.";
  /* ===================== Set All Dictionaries ===================== */
  fl::lib::text::Dictionary tokenDict(FLAGS_tokens_path);
  tokenDict.addEntry(fl::pkg::speech::kBlankToken);
  int blankIdx = tokenDict.getIndex(fl::pkg::speech::kBlankToken);
  int wordSepIdx = networkFlags["wordseparator"] == ""
      ? -1
      : tokenDict.getIndex(networkFlags["wordseparator"]);

  fl::lib::text::LexiconMap lexicon =
      fl::lib::text::loadWords(FLAGS_lexicon_path, -1);
  fl::lib::text::Dictionary wordDict = fl::lib::text::createWordDict(lexicon);
  LOG(INFO)
      << "[Inference tutorial for CTC] Number of classes/tokens in the network: "
      << tokenDict.indexSize();
  LOG(INFO) << "[Inference tutorial for CTC] Number of words in the lexicon: "
            << wordDict.indexSize();
  fl::lib::text::DictionaryMap dicts = {{0, tokenDict}, {1, wordDict}};
  /* ===================== Set LM, Trie, Decoder ===================== */
  int unkWordIdx = wordDict.getIndex(fl::lib::text::kUnkToken);
  auto lm = std::make_shared<fl::lib::text::KenLM>(FLAGS_lm_path, wordDict);
  if (!lm) {
    LOG(FATAL)
        << "[Inference tutorial for CTC] Only KenLM model for language model is supported. "
        << "Failed to load kenlm LM: " << FLAGS_lm_path;
  }
  LOG(INFO) << "[Inference tutorial for CTC] Language model is constructed.";
  std::shared_ptr<fl::lib::text::Trie> trie = fl::pkg::speech::buildTrie(
      "wrd" /* decoderType */,
      true /* useLexicon */,
      lm,
      "max" /* smearing */,
      tokenDict,
      lexicon,
      wordDict,
      wordSepIdx,
      0 /* repLabel */);
  LOG(INFO) << "[Inference tutorial for CTC] Trie is planted.";

  auto decoder = fl::lib::text::LexiconDecoder(
      {.beamSize = FLAGS_beam_size,
       .beamSizeToken = FLAGS_beam_size_token,
       .beamThreshold = FLAGS_beam_threshold,
       .lmWeight = FLAGS_lm_weight,
       .wordScore = FLAGS_word_score,
       .unkScore = -std::numeric_limits<float>::infinity(),
       .silScore = 0,
       .logAdd = false,
       .criterionType = fl::lib::text::CriterionType::CTC},
      trie,
      lm,
      wordSepIdx,
      blankIdx,
      unkWordIdx,
      std::vector<float>(),
      false);
  LOG(INFO) << "[Inference tutorial for CTC] Beam search decoder is created";
  /* ===================== Audio Loading Preparation ===================== */
  fl::lib::audio::FeatureParams featParams(
      FLAGS_sample_rate,
      std::atoll(networkFlags["framesizems"].c_str()),
      std::atoll(networkFlags["framestridems"].c_str()),
      std::atoll(networkFlags["filterbanks"].c_str()),
      std::atoll(networkFlags["lowfreqfilterbank"].c_str()),
      std::atoll(networkFlags["highfreqfilterbank"].c_str()),
      std::atoll(networkFlags["mfcccoeffs"].c_str()),
      fl::pkg::speech::kLifterParam /* lifterparam */,
      std::atoll(networkFlags["devwin"].c_str()) /* delta window */,
      std::atoll(networkFlags["devwin"].c_str()) /* delta-delta window */);
  featParams.useEnergy = false;
  featParams.usePower = false;
  featParams.zeroMeanFrame = false;
  fl::pkg::speech::FeatureType featType;
  if (networkFlags.find("features_type") != networkFlags.end()) {
    featType = fl::pkg::speech::getFeatureType(
                   networkFlags["features_type"], 1, featParams)
                   .second;
  } else {
    // old models TODO remove as per @avidov converting scirpt
    if (networkFlags["pow"] == "true") {
      featType = fl::pkg::speech::FeatureType::POW_SPECTRUM;
    } else if (networkFlags["mfsc"] == "true") {
      featType = fl::pkg::speech::FeatureType::MFSC;
    } else if (networkFlags["mfcc"] == "true") {
      featType = fl::pkg::speech::FeatureType::MFCC;
    } else {
      // raw wave
      featType = fl::pkg::speech::FeatureType::NONE;
    }
  }
  auto inputTransform = fl::pkg::speech::inputFeatures(
      featParams,
      featType,
      {networkFlags["localnrmlleftctx"] == "true",
       networkFlags["localnrmlrightctx"] == "true"},
      /*sfxConf=*/{});
  fl::EditDistanceMeter dst;
  /* ===================== Inference ===================== */
  bool interactive = FLAGS_audio_list == "";
  std::ifstream audioListStream;
  if (!interactive) {
    audioListStream = std::ifstream(FLAGS_audio_list);
  }
  while (true) {
    std::string audioPath;
    if (interactive) {
      LOG(INFO)
          << "[Inference tutorial for CTC]: Waiting the input in the format [audio_path].";
      std::getline(std::cin, audioPath);
    } else {
      if (!std::getline(audioListStream, audioPath)) {
        return 0;
      }
    }
    if (audioPath == "") {
      LOG(INFO)
          << "[Inference tutorial for CTC]: Please provide non-empty input";
      continue;
    }
    if (!fs::exists(audioPath)) {
      LOG(INFO) << "[Inference tutorial for CTC]: File '" << audioPath
                << "' doesn't exist, please provide valid audio path";
      continue;
    }
    auto audioInfo = fl::pkg::speech::loadSoundInfo(audioPath.c_str());
    auto audio = fl::pkg::speech::loadSound<float>(audioPath.c_str());
    fl::Tensor input = inputTransform(
        static_cast<void*>(audio.data()),
        {audioInfo.channels, audioInfo.frames},
        fl::dtype::f32);
    // TODO{fl::Tensor} - consider using a scalar Tensor
    auto inputLen = fl::full({1}, input.dim(0));
    auto rawEmission = fl::pkg::runtime::forwardSequentialModuleWithPadMask(
        fl::input(input), network, inputLen);
    auto emission = rawEmission.tensor().toHostVector<float>();

    const auto& result = decoder.decode(
        emission.data(),
        rawEmission.dim(1) /* time */,
        rawEmission.dim(0) /* ntokens */);

    // Take top hypothesis and cleanup predictions
    auto rawWordPrediction = result[0].words;
    auto rawTokenPrediction = result[0].tokens;

    auto tokenPrediction = fl::pkg::speech::tknPrediction2Ltr(
        rawTokenPrediction,
        tokenDict,
        fl::pkg::speech::kCtcCriterion,
        networkFlags["surround"],
        false /* eostoken */,
        0 /* replabel */,
        false /* usewordpiece */,
        networkFlags["wordseparator"]);
    rawWordPrediction =
        fl::pkg::speech::validateIdx(rawWordPrediction, unkWordIdx);
    auto wordPrediction =
        fl::pkg::speech::wrdIdx2Wrd(rawWordPrediction, wordDict);
    auto wordPredictionStr = fl::lib::join(" ", wordPrediction);
    LOG(INFO) << "[Inference tutorial for CTC]: predicted output for "
              << audioPath << "\n"
              << wordPredictionStr;
  }
  return 0;
}
