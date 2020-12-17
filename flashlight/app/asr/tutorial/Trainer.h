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

namespace fl {
namespace app {
namespace asr {

class Trainer {
 public:
  Trainer(const std::string& mode);

  // Disabling meaningless Trainer copy.
  Trainer(const Trainer&) = delete;
  Trainer& operator=(const Trainer&) = delete;

  void runTraining();
  void trainStep(const std::vector<af::array>& batch);
  void evalStep();

 protected:
  std::shared_ptr<fl::Module> specAug_;
  std::shared_ptr<fl::Module> network_;
  std::shared_ptr<SequenceCriterion> criterion_;
  int64_t epoch_{1};
  int64_t batchIdx_{0};
  double lr_{0};
  int kPadIdx_, kEosIdx_, kUnkIdx_, kMaskIdx_;
  int numFeatures_, numClasses_;
  std::string experimentDirectory_;
  std::string gflagsStr_;
  std::string version_{FL_APP_ASR_VERSION};

  fl::lib::text::LexiconMap lexicon_;
  fl::lib::text::Dictionary tokenDictionary_;
  fl::lib::text::Dictionary wordDictionary_;
  std::hash<std::string> hasher_;

  std::shared_ptr<fl::Dataset> trainDataset_;
  std::unordered_map<std::string, std::shared_ptr<fl::Dataset>> validDatasets_;
  std::unordered_map<std::string, float> bestValidWer_;

  std::shared_ptr<fl::Reducer> reducer_{nullptr};
  std::shared_ptr<fl::FirstOrderOptimizer> networkOptimizer_;

  DatasetMeters trainStatsMeter_;
  std::unordered_map<std::string, DatasetMeters> validStatsMeters_;
  fl::TimeMeter runTimeMeter_;
  fl::TimeMeter batchTimerMeter_{true};
  fl::TimeMeter sampleTimerMeter_{true};
  fl::TimeMeter fwdTimeMeter_{true};
  fl::TimeMeter critFwdTimeMeter_{true};
  fl::TimeMeter bwdTimeMeter_{true};
  fl::TimeMeter optimTimeMeter_{true};
  SpeechStatMeter dataStatsMeter_;

  std::ofstream logWriter_;

  /* Initializers */
  void initTrain();
  void initContinue();
  void initFork();

  void createDictionary();
  void createDatasets();
  void createNetwork();
  void createCriterion();
  void createOptimizer();
  void createSpecAugmentation();

  /* Stateful training helpers */
  void setLr();
  void evalOutput(
      const af::array& output,
      const af::array& target,
      DatasetMeters& mtr);

  /* Stateless training helpers */
  void initArrayFire() const;
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

} // namespace asr
} // namespace app
} // namespace fl
