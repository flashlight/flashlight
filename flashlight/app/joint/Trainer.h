/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdlib>
#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/app/asr/augmentation/SoundEffectConfig.h"
#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/criterion/criterion.h"
#include "flashlight/app/asr/data/FeatureTransforms.h"
#include "flashlight/app/asr/decoder/TranscriptionUtils.h"
#include "flashlight/app/asr/runtime/runtime.h"

#include "flashlight/app/lm/common/Defines.h"
#include "flashlight/app/lm/common/Helpers.h"
#include "flashlight/app/lm/data/TextDataset.h"

#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/common/SequentialBuilder.h"
#include "flashlight/ext/common/Serializer.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"
#include "flashlight/lib/text/tokenizer/PartialFileReader.h"
#include "flashlight/lib/text/tokenizer/Tokenizer.h"

namespace fl {
namespace app {
namespace joint {

#define FL_LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

class Trainer {
 public:
  explicit Trainer(const std::string& mode);

  // Disabling meaningless Trainer copy.
  Trainer(const Trainer&) = delete;
  Trainer& operator=(const Trainer&) = delete;

  void runTraining();
  void trainAsrStep(
      const std::vector<af::array>& batch);
  void trainLmStep(
      const std::vector<af::array>& batch);

  void runEvaluation();
  void evalStep();

  //   void evalLM();

 protected:
  // Shared components
  std::shared_ptr<fl::Module> encoder_; // !!!
  int64_t batchIdx_{0};
  std::string gflagsStr_;
  std::string version_{FL_APP_LM_VERSION};
  int kPadIdx_, kEosIdx_, kUnkIdx_, kMaskIdx_;
  std::hash<std::string> hasher_;
  std::string experimentDirectory_;
  double lr_{0};

  double scaleFactor_;
  unsigned short scaleCounter_;

  // ASR specifics
  int64_t asrEpoch_{1};
  std::shared_ptr<fl::Module> specAug_;
  //   std::shared_ptr<fl::Module> asrFrontEnd_;
  //   std::shared_ptr<fl::Module> asrCriterionLinear_;
  //   std::shared_ptr<fl::app::asr::SequenceCriterion> asrCriterion_;

  fl::lib::text::LexiconMap lexicon_;
  fl::lib::text::Dictionary tokenDictionary_;
  fl::lib::text::Dictionary wordDictionary_;
  int numFeatures_, numClasses_;

  std::shared_ptr<fl::Dataset> asrTrainDataset_;
  std::unordered_map<std::string, std::shared_ptr<fl::Dataset>>
      asrValidDatasets_;
  std::unordered_map<std::string, float> asrBestValidWer_;

  // LM specifics
  int64_t lmEpoch_{1};
  //   std::shared_ptr<fl::Module> lmFrontEnd_;
  //   std::shared_ptr<fl::Module> lmCriterion_;
  std::shared_ptr<fl::app::asr::TransformerCriterion> lmCriterion_;
  fl::lib::text::Dictionary lmDictionary_;

  std::shared_ptr<fl::app::lm::TextDataset> lmTrainDataset_;
  std::unordered_map<std::string, std::shared_ptr<fl::app::lm::TextDataset>>
      lmValidDatasets_;
  std::unordered_map<std::string, float> lmBestValidLoss_;

  // Optimization
  std::shared_ptr<fl::Reducer> reducer_;
  //   std::shared_ptr<fl::FirstOrderOptimizer> optimizer_;
  std::vector<fl::Variable> parameters_;
  std::shared_ptr<fl::FirstOrderOptimizer> ecdOptimizer_;
  std::shared_ptr<fl::FirstOrderOptimizer> critOptimizer_;

  // Meters
  fl::TimeMeter runTimeMeter_;
  fl::TimeMeter asrBatchTimerMeter_{true};
  fl::TimeMeter lmBatchTimerMeter_{true};
  fl::TimeMeter sampleTimerMeter_{true};
  fl::TimeMeter fwdTimeMeter_{true};
  fl::TimeMeter critFwdTimeMeter_{true};
  fl::TimeMeter bwdTimeMeter_{true};
  fl::TimeMeter optimTimeMeter_{true};

  fl::app::asr::DatasetMeters asrTrainStatsMeter_;
  std::unordered_map<std::string, fl::app::asr::DatasetMeters>
      asrValidStatsMeters_;
  fl::app::asr::SpeechStatMeter asrDataStatsMeter_;

  fl::AverageValueMeter lmTrainLossMeter_;
  std::unordered_map<std::string, fl::AverageValueMeter> lmValidLossMeters_;
  fl::AverageValueMeter lmTokenCountMeter_;

  std::ofstream logWriter_;

  /* Initializers */
  void initTrain();
  void initContinue();
  void initFork();
  void initEval();

  void createDictionary();
  void createDatasets();
  void createNetwork();
  void createCriterion();
  void collectParameters();
  void createOptimizer();
  void createSpecAugmentation();

  /* Stateful training helpers */
  std::pair<fl::Variable, fl::Variable> getInputAndTarget(
      const std::vector<af::array>& sample) const;
  void setLr();
  void reduceGrads();
  void evalWer(
      const af::array& output,
      const af::array& target,
      fl::app::asr::DatasetMeters& mtr);

  /* Stateless training helpers */
  void initArrayFire() const;
  std::vector<int> parseCutoffs(int64_t nClasses) const;
  std::pair<std::string, std::string> parseDatasetName(
      const std::string& name) const;
  bool isMaster() const;
  void checkArgs() const;

  /* Meter helpers */
  void resetMeters();
  void syncMeters();
  void stopTimers();

  /* Logging helpers */
  void saveCheckpoint(const std::string& path, const std::string& suffix = "")
      const;
  void logMemoryManagerStatus() const;
  std::string getProgress() const;
};
} // namespace joint
} // namespace app
} // namespace fl
