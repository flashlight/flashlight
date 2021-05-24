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

#include "flashlight/pkg/runtime/Runtime.h"
#include "flashlight/app/lm/common/Defines.h"
#include "flashlight/pkg/text/data/TextDataset.h"

#include "flashlight/pkg/runtime/amp/DynamicScaler.h"
#include "flashlight/fl/distributed/DistributedUtils.h"
#include "flashlight/fl/common/Serializer.h"
#include "flashlight/pkg/runtime/plugin/ModulePlugin.h"
#include "flashlight/fl/contrib/contrib.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"
#include "flashlight/lib/text/tokenizer/PartialFileReader.h"
#include "flashlight/lib/text/tokenizer/Tokenizer.h"

namespace fl {
namespace app {
namespace lm {

#define FL_LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

/* CRITERION OPTIONS */
DECLARE_string(loss_type);
DECLARE_int64(loss_adsm_input_size);
DECLARE_string(loss_adsm_cutoffs);

/* DISTRIBUTED TRAINING */
DECLARE_bool(distributed_enable);
DECLARE_int64(distributed_world_rank);
DECLARE_int64(distributed_world_size);
DECLARE_int64(distributed_max_devices_per_node);
DECLARE_string(distributed_rndv_filepath);

/* RUN OPTIONS */
DECLARE_string(exp_rundir);
DECLARE_string(exp_model_name);
DECLARE_string(exp_init_model_path);

/* DATA OPTIONS */
DECLARE_string(data_dir);
DECLARE_string(data_train);
DECLARE_string(data_valid);
DECLARE_int64(data_batch_size);
DECLARE_int64(data_tokens_per_sample);
DECLARE_string(data_sample_break_mode);
DECLARE_bool(data_use_dynamic_batching);

/* DICTIONARY OPTIONS */
DECLARE_string(dictionary);
DECLARE_int64(dictionary_max_size);

/* TRAIN OPTIONS */
DECLARE_string(train_task);
DECLARE_string(train_arch_dir);
DECLARE_string(train_arch_file);
DECLARE_int64(train_seed);
DECLARE_string(train_optimizer);
DECLARE_int64(train_warmup_updates);
DECLARE_double(train_warmup_init_lr);
DECLARE_double(train_lr);
DECLARE_string(train_lr_schedule);
DECLARE_double(train_momentum);
DECLARE_double(train_weight_decay);
DECLARE_double(train_max_grad_norm);
DECLARE_int64(train_save_updates);
DECLARE_int64(train_report_updates);
DECLARE_int64(train_total_updates);

/* MASK OPTIONS */
DECLARE_double(mask_prob);
DECLARE_double(mask_rand_token_prob);
DECLARE_double(mask_same_token_prob);
DECLARE_int64(mask_min_length);

/* AMP OPTIONS */
DECLARE_bool(fl_amp_use_mixed_precision);
DECLARE_double(fl_amp_scale_factor);
DECLARE_uint64(fl_amp_scale_factor_update_interval);
DECLARE_double(fl_amp_max_scale_factor);
DECLARE_string(fl_optim_mode);

class Trainer {
 public:
  explicit Trainer(const std::string& mode);

  // Disabling meaningless Trainer copy.
  Trainer(const Trainer&) = delete;
  Trainer& operator=(const Trainer&) = delete;

  void runTraining();
  void trainStep();
  void evalStep();
  float runEvaluation();

 protected:
  std::shared_ptr<fl::Module> network_;
  std::shared_ptr<fl::Module> criterion_;
  int64_t epoch_{1};
  int64_t batchIdx_{0};
  float bestLoss_{
      std::numeric_limits<float>::max(),
  };
  int kPadIdx_, kEosIdx_, kUnkIdx_, kMaskIdx_;
  std::string gflagsStr_;
  std::string version_{FL_APP_LM_VERSION};
  std::shared_ptr<fl::ext::DynamicScaler> dynamicScaler;

  fl::lib::text::Dictionary dictionary_;
  std::shared_ptr<TextDataset> trainDataset_;
  std::shared_ptr<TextDataset> validDataset_;

  std::shared_ptr<fl::Reducer> reducer_;
  std::shared_ptr<fl::FirstOrderOptimizer> optimizer_;
  std::vector<fl::Variable> parameters_;

  fl::AverageValueMeter trainLossMeter_;
  fl::AverageValueMeter validLossMeter_;
  fl::TimeMeter runTimeMeter_;
  fl::TimeMeter batchTimerMeter_{true};
  fl::TimeMeter sampleTimerMeter_{true};
  fl::TimeMeter fwdTimeMeter_{true};
  fl::TimeMeter critFwdTimeMeter_{true};
  fl::TimeMeter bwdTimeMeter_{true};
  fl::TimeMeter optimTimeMeter_{true};
  fl::AverageValueMeter tokenCountMeter_;

  std::ofstream logWriter_;

  /* Initializers */
  void initTrain();
  void initContinue();
  void initFork();
  void initEval();

  void createDictionary();
  void createTrainDatasets();
  void createValidDatasets();
  void createNetwork();
  void createCriterion();
  void collectParameters();
  void createOptimizer();

  /* Stateful training helpers */
  std::pair<fl::Variable, fl::Variable> getInputAndTarget(
      const std::vector<af::array>& sample) const;
  void setLr();
  void reduceGrads();

  /* Stateless training helpers */
  void initArrayFire() const;
  std::vector<int> parseCutoffs(int64_t nClasses) const;
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

} // namespace lm
} // namespace app
} // namespace fl
