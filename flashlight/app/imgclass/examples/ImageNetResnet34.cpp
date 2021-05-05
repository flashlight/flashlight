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

#include "flashlight/app/common/Runtime.h"
#include "flashlight/app/imgclass/dataset/Imagenet.h"
#include "flashlight/app/imgclass/examples/Defines.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"
#include "flashlight/ext/image/fl/models/Resnet.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

DEFINE_string(data_dir, "", "Directory of imagenet data");
DEFINE_double(train_lr, 0.1f, "Learning rate");
DEFINE_double(train_momentum, 0.9f, "Momentum");

DEFINE_double(train_wd, 1e-4f, "Weight decay");
DEFINE_uint64(train_epochs, 50, "Number of epochs to train");
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
DEFINE_uint64(data_batch_size, 256, "Total batch size across all gpus");
DEFINE_string(exp_checkpoint_path, "/tmp/model", "Checkpointing prefix path");
DEFINE_int64(exp_checkpoint_epoch, -1, "Checkpoint epoch to load from");
DEFINE_int64(
    exp_log_mem_ops_interval,
    0,
    "Flushes memory manager logs after a specified number of log entries. "
    "1000000 is a reasonable value which will reduces overhead. "
    "Logs when > 0");

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

fl::Variable criterion(const fl::Variable& in, const fl::Variable& target) {
  return categoricalCrossEntropy(logSoftmax(in, 0), target);
}

// Returns the average loss, top 5 error, and top 1 error
std::tuple<double, double, double> evalLoop(
    std::shared_ptr<Sequential> model,
    Dataset& dataset) {
  AverageValueMeter lossMeter;
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);

  // Place the model in eval mode.
  model->eval();
  for (auto& example : dataset) {
    auto inputs = noGrad(example[kImagenetInputIdx]);
    auto output = model->forward(inputs);

    auto target = noGrad(example[kImagenetTargetIdx]);

    // Compute and record the loss.
    auto loss = criterion(output, target);
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
};

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
  af::info();
  if (FLAGS_distributed_enable) {
    fl::ext::initDistributed(
        FLAGS_distributed_world_rank,
        FLAGS_distributed_world_size,
        FLAGS_distributed_max_devices_per_node,
        FLAGS_distributed_rndv_filepath);
  }
  const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();
  const bool isMaster = (worldRank == 0);

  af::setDevice(worldRank);
  af::setSeed(worldSize);
  fl::DynamicBenchmark::setBenchmarkMode(true);

  // log memory manager operations.
  if (FLAGS_exp_log_mem_ops_interval > 0 && isMaster) {
    fl::MemoryManagerInstaller::logIfInstalled(
        lib::pathsConcat(FLAGS_exp_checkpoint_path, "mem.log"),
        FLAGS_exp_log_mem_ops_interval);
  }

  auto reducer =
      std::make_shared<fl::CoalescingReducer>(1.0 / worldSize, true, true);

  if (FLAGS_fl_amp_use_mixed_precision) {
    // Only set the optim mode to O1 if it was left empty
    LOG(INFO) << "Mixed precision training enabled. Will perform loss scaling.";
    auto flOptimLevel = FLAGS_fl_optim_mode.empty()
        ? fl::OptimLevel::DEFAULT
        : fl::OptimMode::toOptimLevel(FLAGS_fl_optim_mode);
    fl::OptimMode::get().setOptimLevel(flOptimLevel);
  }
  unsigned short scaleCounter = 1;
  double scaleFactor =
      FLAGS_fl_amp_use_mixed_precision ? FLAGS_fl_amp_scale_factor : 1.;
  unsigned int kScaleFactorUpdateInterval =
      FLAGS_fl_amp_scale_factor_update_interval;
  double kMaxScaleFactor = FLAGS_fl_amp_max_scale_factor;

  //////////////////////////
  //  Create datasets
  /////////////////////////
  // These are the mean and std for each channel of Imagenet
  const int randomResizeMax = 480;
  const int randomResizeMin = 256;
  const int randomCropSize = 224;
  const float horizontalFlipProb = 0.5f;
  // TransformDataset will apply each transform in a vector to the respective
  // af::array. Thus, we need to `compose` all of the transforms so are each
  // applied only to the image
  ImageTransform trainTransforms = compose(
      {// randomly resize shortest side of image between 256 to 480 for
       // scale invariance
       fl::ext::image::randomResizeTransform(randomResizeMin, randomResizeMax),
       fl::ext::image::randomCropTransform(randomCropSize, randomCropSize),
       fl::ext::image::normalizeImage(
           fl::app::image::kImageNetMean, fl::app::image::kImageNetStd),
       // Randomly flip image with probability of 0.5
       fl::ext::image::randomHorizontalFlipTransform(horizontalFlipProb)});
  ImageTransform valTransforms = compose(
      {// Resize shortest side to 256, then take a center crop
       fl::ext::image::resizeTransform(randomResizeMin),
       fl::ext::image::centerCropTransform(randomCropSize),
       fl::ext::image::normalizeImage(
           fl::app::image::kImageNetMean, fl::app::image::kImageNetStd)});

  const int64_t batchSizePerGpu = FLAGS_data_batch_size;
  const int64_t prefetchThreads = 10;
  const int64_t prefetchSize = FLAGS_data_batch_size;
  auto labelMap = getImagenetLabels(labelPath);
  auto trainDataset = fl::ext::image::DistributedDataset(
      imagenetDataset(trainList, labelMap, {trainTransforms}),
      worldRank,
      worldSize,
      batchSizePerGpu,
      1, // train_n_repeatedaug
      prefetchThreads,
      prefetchSize,
      fl::BatchDatasetPolicy::SKIP_LAST);

  auto valDataset = fl::ext::image::DistributedDataset(
      imagenetDataset(valList, labelMap, {valTransforms}),
      worldRank,
      worldSize,
      batchSizePerGpu,
      1, // train_n_repeatedaug
      prefetchThreads,
      prefetchSize,
      fl::BatchDatasetPolicy::INCLUDE_LAST);

  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  auto model = fl::ext::image::resnet34();
  // synchronize parameters of the model so that the parameters in each process
  // is the same
  fl::allReduceParameters(model);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(model, reducer);

  SGDOptimizer opt(
      model->params(), FLAGS_train_lr, FLAGS_train_momentum, FLAGS_train_wd);

  auto lrScheduler = [&opt](int epoch) {
    // Adjust learning rate every 30 epoch after 30
    if (epoch == 60 || epoch == 90 || epoch == 120) {
      const float newLr = opt.getLr() * 0.1;
      LOG(INFO) << "Setting learning rate to: " << newLr;
      opt.setLr(newLr);
    }
  };

  // Small utility functions to load and save models
  auto saveModel = [&model, &isMaster](int epoch) {
    if (isMaster) {
      std::string modelPath = FLAGS_exp_checkpoint_path + std::to_string(epoch);
      LOG(INFO) << "Saving model to file: " << modelPath;
      fl::save(modelPath, model);
    }
  };

  auto loadModel = [&model](int epoch) {
    std::string modelPath = FLAGS_exp_checkpoint_path + std::to_string(epoch);
    LOG(INFO) << "Loading model from file: " << modelPath;
    fl::load(modelPath, model);
  };
  if (FLAGS_exp_checkpoint_epoch >= 0) {
    loadModel(FLAGS_exp_checkpoint_epoch);
  }

  // The main training loop
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);
  AverageValueMeter trainLossMeter;

  fl::TimeMeter sampleTimerMeter{true};
  fl::TimeMeter fwdTimeMeter{true};
  fl::TimeMeter critFwdTimeMeter{true};
  fl::TimeMeter bwdTimeMeter{true};
  fl::TimeMeter optimTimeMeter{true};
  fl::TimeMeter timeMeter{true};

  for (int epoch = (FLAGS_exp_checkpoint_epoch + 1); epoch < FLAGS_train_epochs;
       epoch++) {
    trainDataset.resample(epoch);
    lrScheduler(epoch);

    // Get an iterator over the data
    int idx = 0;
    for (auto& example : trainDataset) {
      timeMeter.resume();
      opt.zeroGrad();

      // Make Variables from the input arrays.
      sampleTimerMeter.resume();
      auto inputs = noGrad(example[kImagenetInputIdx]);
      if (FLAGS_fl_amp_use_mixed_precision && FLAGS_fl_optim_mode.empty()) {
        // In case AMP is activated with DEFAULT mode,
        // we manually cast input to fp16.
        inputs = inputs.as(f16);
      }
      auto target = noGrad(example[kImagenetTargetIdx]);
      af::sync();
      sampleTimerMeter.stopAndIncUnit();

      // Get the activations from the model.
      fwdTimeMeter.resume();
      auto output = model->forward(inputs);
      af::sync();
      fwdTimeMeter.stopAndIncUnit();

      // Compute and record the loss.
      critFwdTimeMeter.resume();
      auto loss = criterion(output, target);
      af::sync();
      critFwdTimeMeter.stopAndIncUnit();

      if (FLAGS_fl_amp_use_mixed_precision) {
        ++scaleCounter;
        loss = loss * scaleFactor;

        auto scaledLoss = loss.array() / worldSize;
        if (FLAGS_distributed_enable) {
          fl::allReduce(scaledLoss);
        }
        if (fl::app::isInvalidArray(scaledLoss)) {
          FL_LOG(INFO) << "Loss has NaN values in 2, in proc: "
                       << fl::getWorldRank();
          scaleFactor = scaleFactor / 2.0f;
          FL_LOG(INFO) << "AMP: Scale factor decreased (loss). New value :\t "
                       << scaleFactor;
          continue;
        }
      }

      trainLossMeter.add(loss.array());
      top5Acc.add(output.array(), target.array());
      top1Acc.add(output.array(), target.array());

      // Backprop, update the weights and then zero the gradients.
      bwdTimeMeter.resume();
      loss.backward();

      if (FLAGS_distributed_enable) {
        reducer->finalize();
      }
      af::sync();
      bwdTimeMeter.stopAndIncUnit();

      optimTimeMeter.resume();
      auto retrySample = false;
      if (FLAGS_fl_amp_use_mixed_precision) {
        for (auto& p : model->params()) {
          p.grad() = p.grad() / scaleFactor;
          if (fl::app::isInvalidArray(p.grad().array())) {
            FL_LOG(INFO) << "Grad has NaN values in 3, in proc: "
                         << fl::getWorldRank();
            if (scaleFactor >= fl::kAmpMinimumScaleFactorValue) {
              scaleFactor = scaleFactor / 2.0f;
              FL_LOG(INFO) << "AMP: Scale factor decreased (grad). New value:\t"
                           << scaleFactor;
              retrySample = true;
            } else {
              FL_LOG(FATAL)
                  << "Minimum loss scale reached: "
                  << fl::kAmpMinimumScaleFactorValue
                  << " with over/underflowing gradients. Lowering the "
                  << "learning rate, using gradient clipping, or "
                  << "increasing the batch size can help resolve "
                  << "loss explosion.";
            }
            scaleCounter = 1;
            break;
          }
        }
      }
      if (retrySample) {
        timeMeter.stopAndIncUnit();
        optimTimeMeter.stopAndIncUnit();
        continue;
      }

      opt.step();
      af::sync();
      optimTimeMeter.stopAndIncUnit();
      timeMeter.stopAndIncUnit();

      // Compute and record the prediction error.
      double trainLoss = trainLossMeter.value()[0];
      if (++idx % 50 == 0) {
        fl::ext::syncMeter(trainLossMeter);
        fl::ext::syncMeter(timeMeter);
        fl::ext::syncMeter(top5Acc);
        fl::ext::syncMeter(top1Acc);
        double time = timeMeter.value();
        double samplePerSecond = FLAGS_data_batch_size * worldSize / time;
        FL_LOG_MASTER(INFO)
            << "Epoch " << epoch << std::setprecision(5) << " Batch: " << idx
            << " Samples per second " << samplePerSecond
            << " : Total Time(ms): " << fl::lib::format("%.2f", time * 1000)
            << " : Sample Time(ms): "
            << fl::lib::format("%.2f", sampleTimerMeter.value() * 1000)
            << " : Forward Time(ms): "
            << fl::lib::format("%.2f", fwdTimeMeter.value() * 1000)
            << " : Criterion Forward Time(ms): "
            << fl::lib::format("%.2f", critFwdTimeMeter.value() * 1000)
            << " : Backward Time(ms): "
            << fl::lib::format("%.2f", bwdTimeMeter.value() * 1000)
            << " : Optim Time(ms): "
            << fl::lib::format("%.2f", optimTimeMeter.value() * 1000)
            << ": Avg Train Loss: " << trainLoss
            << ": Train Top5 Accuracy( %): " << top5Acc.value()
            << ": Train Top1 Accuracy( %): " << top1Acc.value();
        top5Acc.reset();
        top1Acc.reset();
        trainLossMeter.reset();
        timeMeter.reset();
        sampleTimerMeter.reset();
        fwdTimeMeter.reset();
        critFwdTimeMeter.reset();
        bwdTimeMeter.reset();
        optimTimeMeter.reset();
      }

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
    }
    timeMeter.reset();
    timeMeter.stop();

    double valLoss, valTop1Error, valTop5Err;
    std::tie(valLoss, valTop5Err, valTop1Error) = evalLoop(model, valDataset);

    FL_LOG_MASTER(INFO) << "Epoch " << epoch << std::setprecision(5)
                        << " Validation Loss: " << valLoss
                        << " Validation Top5 Error (%): " << valTop5Err
                        << " Validation Top1 Error (%): " << valTop1Error;
    saveModel(epoch);
  }
  FL_LOG_MASTER(INFO) << "Training complete";
}
