/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <exception>
#include <iomanip>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/app/imgclass/examples/Defines.h"
#include "flashlight/ext/amp/DynamicScaler.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/pkg/vision/dataset/Transforms.h"
#include "flashlight/pkg/vision/dataset/DistributedDataset.h"
#include "flashlight/pkg/vision/models/ViT.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/lib/common/BetaDistribution.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/pkg/runtime/Runtime.h"
#include "flashlight/pkg/vision/dataset/Imagenet.h"

DEFINE_string(data_dir, "", "Directory of imagenet data");
DEFINE_uint64(data_batch_size, 64, "Batch size per gpus");
DEFINE_uint64(data_prefetch_thread, 10, "Batch size per gpus");

DEFINE_double(train_lr, 1e-3, "Learning rate");
DEFINE_double(train_warmup_epochs, 5, "Number of epochs to warmup");
DEFINE_double(train_beta1, 0.9, "Adam beta1");
DEFINE_double(train_beta2, 0.999, "Adam beta2");
DEFINE_double(train_wd, 5e-2, "Weight decay");
DEFINE_uint64(train_epochs, 300, "Number of epochs to train");
DEFINE_double(train_dropout, 0., "Dropout");
DEFINE_double(train_layerdrop, 0.1, "Layer drop");
DEFINE_double(train_maxgradnorm, 0., "Maximum gradient norm");
DEFINE_int64(train_seed, 1, "Seed");

DEFINE_double(train_aug_p_label_smoothing, 0.1, "Label smoothing probability");
DEFINE_double(train_aug_p_randomerase, 0.25, "Random erasing probablity");
DEFINE_double(
    train_aug_p_randomeaug,
    0.5,
    "Probablity of applying random augentation transform to each sample");
DEFINE_uint64(
    train_aug_n_randomeaug,
    2,
    "Number of random augentation transforms applied to each sample");
DEFINE_uint64(
    train_aug_n_repeatedaug,
    3,
    "Number of repetitions created for each sample");
DEFINE_bool(train_aug_use_mix, true, "Enable mixup and cutmix in training");
DEFINE_double(
    train_aug_p_mixup,
    0.8,
    "Alpha of mixup. Mixup is disabled with 0");
DEFINE_double(
    train_aug_p_cutmix,
    1.0,
    "Alpha of cutmix. Cutmix is disabled with 0");
DEFINE_double(
    train_aug_p_switchmix,
    0.5,
    "Probability of switching between cutmix and mixup");

DEFINE_bool(distributed_enable, true, "Enable distributed training");
DEFINE_int64(
    distributed_max_devices_per_node,
    8,
    "the maximum number of devices per training node");
DEFINE_int64(
    distributed_world_rank,
    0,
    "rank of the process (Used if distributed_rndv_filepath is not empty)");
DEFINE_int64(
    distributed_world_size,
    1,
    "total number of the process (Used if distributed_rndv_filepath is not empty)");
DEFINE_string(
    distributed_rndv_filepath,
    "",
    "Shared file path used for setting up rendezvous."
    "If empty, uses MPI to initialize.");

DEFINE_string(exp_checkpoint_path, "/tmp/model", "Checkpointing prefix path");
DEFINE_int64(exp_checkpoint_epoch, -1, "Checkpoint epoch to load from");

DEFINE_int64(model_layers, 12, "Number of transformer layers");
DEFINE_int64(
    model_hidden_emb_size,
    768,
    "Hidden embedding size of transformer block");
DEFINE_int64(model_mlp_size, 3072, "Mlp size of transformer block");
DEFINE_int64(model_heads, 12, "Number of heads of transformer block");

DEFINE_bool(
    fl_amp_use_mixed_precision,
    false,
    "[train] Use mixed precision for training - scale loss and gradients up and down "
    "by a scale factor that changes over time. If no fl optim mode is "
    "specified with --fl_optim_mode when passing this flag, automatically "
    "sets the optim mode to O1.");
DEFINE_double(
    fl_amp_scale_factor,
    65536.,
    "[train] Starting scale factor to use for loss scaling "
    " with mixed precision training");
DEFINE_uint64(
    fl_amp_scale_factor_update_interval,
    2000,
    "[train] Update interval for adjusting loss scaling in mixed precision training");
DEFINE_double(
    fl_amp_max_scale_factor,
    65536.,
    "[train] Maximum value for the loss scale factor in mixed precision training");
DEFINE_string(
    fl_optim_mode,
    "",
    "[train] Sets the flashlight optimization mode. "
    "Optim modes can be O1, O2, or O3.");

using namespace fl;
using fl::ext::image::compose;
using fl::ext::image::ImageTransform;
using namespace fl::app::imgclass;

#define FL_LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

// Returns the average loss, top 5 error, and top 1 error
std::tuple<double, double, double> evalLoop(
    std::shared_ptr<fl::ext::image::ViT> model,
    Dataset& dataset) {
  AverageValueMeter lossMeter;
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);

  // Place the model in eval mode.
  model->eval();
  for (auto& example : dataset) {
    auto inputs = noGrad(example[kImagenetInputIdx]);
    auto output = model->forward({inputs}).front();
    output = logSoftmax(output, 0).as(output.type());

    auto target = noGrad(example[kImagenetTargetIdx]);

    // Compute and record the loss.
    auto loss = categoricalCrossEntropy(output, target);
    lossMeter.add(loss.array().scalar<float>());
    top5Acc.add(output.array(), target.array());
    top1Acc.add(output.array(), target.array());
  }
  model->train();
  fl::ext::syncMeter(lossMeter);
  fl::ext::syncMeter(top5Acc);
  fl::ext::syncMeter(top1Acc);

  double loss = lossMeter.value()[0];
  return std::make_tuple(loss, top5Acc.value(), top1Acc.value());
}

int main(int argc, char** argv) {
  fl::init();
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::string labelPath = lib::pathsConcat(FLAGS_data_dir, "labels.txt");
  const std::string trainList = lib::pathsConcat(FLAGS_data_dir, "train");
  const std::string valList = lib::pathsConcat(FLAGS_data_dir, "val");

  /////////////////////////
  // Setup distributed training
  ////////////////////////
  std::shared_ptr<fl::Reducer> reducer;
  if (FLAGS_distributed_enable) {
    fl::ext::initDistributed(
        FLAGS_distributed_world_rank,
        FLAGS_distributed_world_size,
        FLAGS_distributed_max_devices_per_node,
        FLAGS_distributed_rndv_filepath);
    reducer = std::make_shared<fl::CoalescingReducer>(
        1.0 / fl::getWorldSize(), true, true);
  }
  af::info();
  FL_LOG_MASTER(INFO) << "Gflags after parsing \n"
                      << fl::app::serializeGflags("; ");

  const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();
  const bool isMaster = (worldRank == 0);

  //////////////////////////
  //  Create datasets
  /////////////////////////
  FL_LOG_MASTER(INFO) << "Creating dataset";

  // TODO: only support training with image shape 224 x 224 in this example
  const int imageSize = 224;
  // Conventional image resize parameter used for evaluation
  const int randomResizeMin = imageSize / .875;
  auto fillImg = af::tile(
      af::array(1, 1, 3, 1, fl::app::image::kImageNetMean.data()),
      imageSize,
      imageSize);

  ImageTransform trainTransforms = compose(
      {fl::ext::image::randomResizeCropTransform(
           imageSize,
           0.08, // scaleLow
           1.0, // scaleHigh
           3. / 4., // ratioLow
           4. / 3. // ratioHigh
           ),
       fl::ext::image::randomHorizontalFlipTransform(0.5 // flipping probablity
                                                     ),
       fl::ext::image::randomAugmentationDeitTransform(
           FLAGS_train_aug_p_randomeaug, FLAGS_train_aug_n_randomeaug, fillImg),
       fl::ext::image::normalizeImage(
           fl::app::image::kImageNetMean, fl::app::image::kImageNetStd),
       fl::ext::image::randomEraseTransform(FLAGS_train_aug_p_randomerase)});

  ImageTransform valTransforms = compose(
      {fl::ext::image::resizeTransform(randomResizeMin),
       fl::ext::image::centerCropTransform(imageSize),
       fl::ext::image::normalizeImage(
           fl::app::image::kImageNetMean, fl::app::image::kImageNetStd)});

  const int64_t prefetchSize = FLAGS_data_batch_size * 10;
  auto labelMap = getImagenetLabels(labelPath);
  if (labelMap.size() != fl::app::image::kNumImageNetClasses) {
    FL_LOG_MASTER(INFO)
        << "[Warning] You are not using all ImageNet classes (1000)";
  }

  auto trainDataset = std::make_shared<fl::ext::image::DistributedDataset>(
      imagenetDataset(trainList, labelMap, {trainTransforms}),
      worldRank,
      worldSize,
      FLAGS_data_batch_size,
      FLAGS_train_aug_n_repeatedaug,
      FLAGS_data_prefetch_thread,
      prefetchSize,
      fl::BatchDatasetPolicy::SKIP_LAST);
  FL_LOG_MASTER(INFO) << "[trainDataset size] " << trainDataset->size();

  auto valDataset = fl::ext::image::DistributedDataset(
      imagenetDataset(valList, labelMap, {valTransforms}),
      worldRank,
      worldSize,
      FLAGS_data_batch_size,
      1, // train_n_repeatedaug
      FLAGS_data_prefetch_thread,
      prefetchSize,
      fl::BatchDatasetPolicy::INCLUDE_LAST);
  FL_LOG_MASTER(INFO) << "[valDataset size] " << valDataset.size();

  //////////////////////////
  //  Create model
  /////////////////////////
  fl::setSeed(FLAGS_train_seed); // Making sure the models are initialized in
                                 // the same way across different processes
  auto model = std::make_shared<fl::ext::image::ViT>(
      FLAGS_model_layers,
      FLAGS_model_hidden_emb_size,
      FLAGS_model_mlp_size,
      FLAGS_model_heads,
      FLAGS_train_dropout,
      FLAGS_train_layerdrop,
      labelMap.size());
  FL_LOG_MASTER(INFO) << "[model with parameters " << fl::numTotalParams(model)
                      << "] " << model->prettyString();

  if (FLAGS_distributed_enable) {
    fl::allReduceParameters(model);
    fl::distributeModuleGrads(model, reducer);
  }

  // Setting different seeds for better randomness
  fl::setSeed(worldRank + FLAGS_train_seed);
  std::srand(worldRank + FLAGS_train_seed);
  fl::DynamicBenchmark::setBenchmarkMode(true);

  //////////////////////////
  //  Optimizer
  /////////////////////////
  auto modelParams = model->params();
  std::vector<fl::Variable> paramsWithWeightDecay(
      modelParams.begin() + 2, modelParams.end());
  auto optWithWeightDecay = AdamOptimizer(
      paramsWithWeightDecay,
      FLAGS_train_lr,
      FLAGS_train_beta1,
      FLAGS_train_beta2,
      1e-8,
      FLAGS_train_wd);

  // Excluding class token and positional embedding from weight decay
  std::vector<fl::Variable> paramsNoWeightDecay(
      modelParams.begin(), modelParams.begin() + 2);
  auto optNoWeightDecay = AdamOptimizer(
      paramsNoWeightDecay,
      FLAGS_train_lr,
      FLAGS_train_beta1,
      FLAGS_train_beta2,
      1e-8,
      0);

  //////////////////////////
  //  Small utility functions
  /////////////////////////
  auto lrScheduler = [&optWithWeightDecay, &optNoWeightDecay](int epoch) {
    // following https://git.io/JYOOV
    double lr;
    if (epoch <= FLAGS_train_warmup_epochs) {
      lr = (epoch - 1) * FLAGS_train_lr / FLAGS_train_warmup_epochs;
      lr = std::max(lr, 1e-6);
    } else {
      lr = 1e-5 +
          0.5 * (FLAGS_train_lr - 1e-5) *
              (std::cos(
                   ((double)epoch - 1) / ((double)FLAGS_train_epochs) * M_PI) +
               1);
    }
    optWithWeightDecay.setLr(lr);
    optNoWeightDecay.setLr(lr);
  };

  int batchIdx = 0;
  int epoch = 0;
  auto saveModel =
      [&model, &isMaster, &epoch, &batchIdx](const std::string& suffix = "") {
        if (isMaster) {
          std::string modelPath = FLAGS_exp_checkpoint_path + suffix;
          FL_LOG_MASTER(INFO) << "Saving model to file: " << modelPath;
          fl::save(modelPath, model, batchIdx, epoch);
        }
      };
  auto loadModel = [&model, &epoch, &batchIdx]() {
    FL_LOG_MASTER(INFO) << "Loading model from file: "
                        << FLAGS_exp_checkpoint_path;
    fl::load(FLAGS_exp_checkpoint_path, model, batchIdx, epoch);
  };
  if (FLAGS_exp_checkpoint_epoch >= 0) {
    loadModel();
  }

  auto betaGeneratorMixup = fl::lib::beta_distribution<float>(
      FLAGS_train_aug_p_mixup, FLAGS_train_aug_p_mixup);
  auto betaGeneratorCutmix = fl::lib::beta_distribution<float>(
      FLAGS_train_aug_p_cutmix, FLAGS_train_aug_p_cutmix);
  std::mt19937_64 engine(worldRank);

  //////////////////////////
  // The main training loop
  /////////////////////////
  std::shared_ptr<fl::ext::DynamicScaler> dynamicScaler;
  if (FLAGS_fl_amp_use_mixed_precision) {
    FL_LOG_MASTER(INFO)
        << "Mixed precision training enabled. Will perform loss scaling.";
    auto flOptimLevel = FLAGS_fl_optim_mode.empty()
        ? fl::OptimLevel::DEFAULT
        : fl::OptimMode::toOptimLevel(FLAGS_fl_optim_mode);
    fl::OptimMode::get().setOptimLevel(flOptimLevel);

    dynamicScaler = std::make_shared<fl::ext::DynamicScaler>(
        FLAGS_fl_amp_scale_factor,
        FLAGS_fl_amp_max_scale_factor,
        FLAGS_fl_amp_scale_factor_update_interval);
  }

  fl::TimeMeter sampleTimerMeter{true};
  fl::TimeMeter fwdTimeMeter{true};
  fl::TimeMeter critFwdTimeMeter{true};
  fl::TimeMeter bwdTimeMeter{true};
  fl::TimeMeter optimTimeMeter{true};

  FL_LOG_MASTER(INFO) << "[Training starts]";
  TimeMeter timeMeter;
  AverageValueMeter trainLossMeter;
  for (; epoch < FLAGS_train_epochs; epoch++) {
    trainDataset->resample(epoch);
    lrScheduler(epoch);

    timeMeter.resume();
    for (int idx = 0; idx < trainDataset->size(); idx++, batchIdx++) {
      // 1. Sample
      sampleTimerMeter.resume();
      auto sample = trainDataset->get(idx);
      auto inputArray = sample[kImagenetInputIdx];
      auto targetArray = sample[kImagenetTargetIdx];

      // Mixup + Cutmix + label smoothing
      if (FLAGS_train_aug_use_mix) {
        float mixP =
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);

        if (mixP > FLAGS_train_aug_p_switchmix) {
          // using mixup
          float lambda = betaGeneratorMixup(engine);
          std::tie(inputArray, targetArray) = fl::ext::image::mixupBatch(
              lambda,
              inputArray,
              targetArray,
              fl::app::image::kNumImageNetClasses,
              FLAGS_train_aug_p_label_smoothing);
        } else {
          // using cutmix
          float lambda = betaGeneratorCutmix(engine);
          std::tie(inputArray, targetArray) = fl::ext::image::cutmixBatch(
              lambda,
              inputArray,
              targetArray,
              fl::app::image::kNumImageNetClasses,
              FLAGS_train_aug_p_label_smoothing);
        }
      } else {
        targetArray = fl::ext::image::oneHot(
            targetArray,
            fl::app::image::kNumImageNetClasses,
            FLAGS_train_aug_p_label_smoothing);
      }
      auto input = noGrad(inputArray);
      auto target = noGrad(targetArray);
      if (FLAGS_fl_amp_use_mixed_precision && FLAGS_fl_optim_mode.empty()) {
        // In case AMP is activated with DEFAULT mode,
        // we manually cast input to fp16.
        input = input.as(f16);
      }

      fl::sync();
      sampleTimerMeter.stopAndIncUnit();

      while (true) {
        // 2. Forward
        fwdTimeMeter.resume();
        auto output = model->forward({input}).front();
        output = logSoftmax(output, 0).as(output.type());

        critFwdTimeMeter.resume();
        auto y = target.as(output.type());
        auto loss = fl::mean(fl::sum(fl::negate(y * output), {0}), {1});
        fl::sync();
        fwdTimeMeter.stopAndIncUnit();
        critFwdTimeMeter.stopAndIncUnit();

        // 3. Backward
        bwdTimeMeter.resume();
        optWithWeightDecay.zeroGrad();
        optNoWeightDecay.zeroGrad();
        bool scaleIsValid = fl::app::backwardWithScaling(
            loss, modelParams, dynamicScaler, reducer);
        fl::sync();
        bwdTimeMeter.stopAndIncUnit();
        if (!scaleIsValid) {
          continue;
        }
        trainLossMeter.add(loss.array());
        break;
      }

      // 4. Optimize
      optimTimeMeter.resume();
      if (FLAGS_train_maxgradnorm > 0) {
        fl::clipGradNorm(modelParams, FLAGS_train_maxgradnorm);
      }
      optWithWeightDecay.step();
      optNoWeightDecay.step();
      optimTimeMeter.stopAndIncUnit();

      // 5. Compute and record the prediction error.
      int interval = 100;
      if (idx && idx % interval == 0) {
        timeMeter.stop();
        fl::ext::syncMeter(trainLossMeter);
        fl::ext::syncMeter(timeMeter);
        double time = timeMeter.value() / interval;
        double samplePerSecond = FLAGS_data_batch_size * worldSize / time;
        FL_LOG_MASTER(INFO)
            << "Epoch " << epoch << std::setprecision(5) << " Batch: " << idx
            << " Throughput " << samplePerSecond
            << " | : Batch time(ms): " << fl::lib::format("%.2f", time * 1000)
            << " : Sample Time(ms): "
            << fl::lib::format("%.2f", sampleTimerMeter.value() * 1000)
            << " : Forward Time(ms): "
            << fl::lib::format("%.2f", fwdTimeMeter.value() * 1000)
            << " : Criterion Forward Time(ms): "
            << fl::lib::format("%.2f", critFwdTimeMeter.value() * 1000)
            << " : Backward Time(ms): "
            << fl::lib::format("%.2f", bwdTimeMeter.value() * 1000)
            << " : Optimization Time(ms): "
            << fl::lib::format("%.2f", optimTimeMeter.value() * 1000)
            << " | LR: " << optNoWeightDecay.getLr()
            << ": Avg Train Loss: " << trainLossMeter.value()[0];
        trainLossMeter.reset();
        timeMeter.reset();
        timeMeter.resume();
      }
    }
    timeMeter.stop();
    timeMeter.reset();
    sampleTimerMeter.reset();
    fwdTimeMeter.reset();
    critFwdTimeMeter.reset();
    bwdTimeMeter.reset();
    optimTimeMeter.reset();

    double valLoss, valTop1Error, valTop5Err;
    std::tie(valLoss, valTop5Err, valTop1Error) = evalLoop(model, valDataset);

    FL_LOG_MASTER(INFO) << "Epoch " << epoch << std::setprecision(5)
                        << " Validation Loss: " << valLoss
                        << " Validation Top5 Error (%): " << valTop5Err
                        << " Validation Top1 Error (%): " << valTop1Error;
    saveModel();
  }
  FL_LOG_MASTER(INFO) << "Training complete";

  return 0;
}
