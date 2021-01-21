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

#include "flashlight/app/imgclass/dataset/Imagenet.h"
#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"
#include "flashlight/ext/image/fl/models/Resnet.h"
#include "flashlight/ext/image/fl/models/ViT.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/lib/common/System.h"

DEFINE_string(data_dir, "", "Directory of imagenet data");
DEFINE_uint64(data_batch_size, 256, "Batch size per gpus");

DEFINE_double(train_lr, 5e-4, "Learning rate");
DEFINE_double(train_warmup_updates, 10000, "train_warmup_updates");
DEFINE_double(train_beta1, 0.9, "Learning rate");
DEFINE_double(train_beta2, 0.999, "Learning rate");
DEFINE_double(train_wd, 5e-2, "Weight decay");
DEFINE_uint64(train_epochs, 50, "Number of epochs to train");
DEFINE_double(train_dropout, 0., "Number of epochs to train");
DEFINE_double(train_layerdrop, 0., "Number of epochs to train");
DEFINE_double(train_maxgradnorm, 1., "");

DEFINE_double(train_p_randomerase, 1., "");
DEFINE_double(train_p_mixup, 0., "");

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

DEFINE_int64(model_layers, 12, "model_layers");
DEFINE_int64(model_hidden_emb_size, 768, "model_hidden_size");
DEFINE_int64(model_mlp_size, 3072, "model_mlp_size");
DEFINE_int64(model_heads, 12, "model_heads");

using namespace fl;
using fl::ext::image::compose;
using fl::ext::image::ImageTransform;
using namespace fl::app::imgclass;

#define FL_LOG_MASTER(lvl) LOG_IF(lvl, (fl::getWorldRank() == 0))

const double pi = std::acos(-1);

// Returns the average loss, top 5 error, and top 1 error
std::tuple<double, double, double> evalLoop(
    std::shared_ptr<Container> model,
    Dataset& dataset) {
  AverageValueMeter lossMeter;
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);

  // Place the model in eval mode.
  model->eval();
  for (auto& example : dataset) {
    auto inputs = noGrad(example[kImagenetInputIdx]);
    auto output = model->forward({inputs}).front();

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

  af::setSeed(worldSize);

  auto reducer =
      std::make_shared<fl::CoalescingReducer>(1.0 / worldSize, true, true);

  //////////////////////////
  //  Create datasets
  /////////////////////////
  FL_LOG_MASTER(INFO) << "Creating dataset";
  // These are the mean and std for each channel of Imagenet
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
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
       fl::ext::image::randomEraseTransform(FLAGS_train_p_randomerase),
       fl::ext::image::normalizeImage(mean, std),
       // Randomly flip image with probability of 0.5
       fl::ext::image::randomHorizontalFlipTransform(horizontalFlipProb)});
  ImageTransform valTransforms =
      compose({// Resize shortest side to 256, then take a center crop
               fl::ext::image::resizeTransform(randomResizeMin),
               fl::ext::image::centerCropTransform(randomCropSize),
               fl::ext::image::normalizeImage(mean, std)});

  const int64_t batchSizePerGpu = FLAGS_data_batch_size;
  const int64_t prefetchThreads = 10;
  const int64_t prefetchSize = FLAGS_data_batch_size;
  auto labelMap = getImagenetLabels(labelPath);
  auto trainDataset = std::make_shared<fl::ext::image::DistributedDataset>(
      imagenetDataset(trainList, labelMap, {trainTransforms}),
      worldRank,
      worldSize,
      batchSizePerGpu,
      prefetchThreads,
      prefetchSize,
      fl::BatchDatasetPolicy::SKIP_LAST);
  FL_LOG_MASTER(INFO) << "[trainDataset size] " << trainDataset->size();

  auto trainDataset1 = std::make_shared<fl::ext::image::DistributedDataset>(
      imagenetDataset(trainList, labelMap, {trainTransforms}),
      worldRank,
      worldSize,
      batchSizePerGpu,
      prefetchThreads,
      prefetchSize,
      fl::BatchDatasetPolicy::SKIP_LAST,
      4399);

  auto valDataset = fl::ext::image::DistributedDataset(
      imagenetDataset(valList, labelMap, {valTransforms}),
      worldRank,
      worldSize,
      batchSizePerGpu,
      prefetchThreads,
      prefetchSize,
      fl::BatchDatasetPolicy::INCLUDE_LAST);
  FL_LOG_MASTER(INFO) << "[valDataset size] " << valDataset.size();

  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  auto model = std::make_shared<fl::ext::image::ViT>(
      FLAGS_model_layers,
      FLAGS_model_hidden_emb_size,
      FLAGS_model_mlp_size,
      FLAGS_model_heads,
      FLAGS_train_dropout,
      FLAGS_train_layerdrop,
      1000);
  FL_LOG_MASTER(INFO) << "[model] " << model->prettyString();
  // synchronize parameters of the model so that the parameters in each process
  // is the same
  fl::allReduceParameters(model);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(model, reducer);

  auto opt = AdamOptimizer(
      model->params(),
      FLAGS_train_lr,
      FLAGS_train_beta1,
      FLAGS_train_beta2,
      1e-8,
      FLAGS_train_wd);

  auto lrScheduler = [&opt](int epoch, int update) {
    double lr = FLAGS_train_lr;
    if (update < FLAGS_train_warmup_updates) {
      // warmup stage
      lr = 1e-7 +
          (FLAGS_train_lr - 1e-7) * update /
              (double(FLAGS_train_warmup_updates));
    } else {
      // cosine decay
      lr = FLAGS_train_lr *
          std::cos(((double)epoch) / ((double)FLAGS_train_epochs) * pi / 2.0);
    }
    opt.setLr(lr);
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
  FL_LOG_MASTER(INFO) << "[Training starts]";
  TimeMeter timeMeter;
  TopKMeter top5Acc(5);
  TopKMeter top1Acc(1);
  AverageValueMeter trainLossMeter;
  int batchIdx = 0;
  for (int epoch = (FLAGS_exp_checkpoint_epoch + 1); epoch < FLAGS_train_epochs;
       epoch++) {
    trainDataset->resample(epoch);
    trainDataset1->resample(epoch + 4399);

    // Get an iterator over the data
    timeMeter.resume();
    int idx = 0;
    // for (auto& example : trainDataset) {
    for (; idx < trainDataset->size();) {
      auto example = trainDataset->get(idx);
      auto example1 = trainDataset1->get(idx);
      lrScheduler(epoch, batchIdx++);
      opt.zeroGrad();

      // Make a Variable from the input array.
      Variable inputs;
      float mixP =
          static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
      float lambda =
          static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
      float example1Weight = FLAGS_train_p_mixup;
      if (FLAGS_train_p_mixup > 0 && mixP > 0.5) {
        // mixup
        inputs = FLAGS_train_p_mixup * noGrad(example1[kImagenetInputIdx]) +
            (1 - FLAGS_train_p_mixup) * noGrad(example[kImagenetInputIdx]);
      } else {
        // cut mix
        auto in = example[kImagenetInputIdx];
        const int w = in.dims(0);
        const int h = in.dims(1);
        const int maskW = std::round(w * lambda);
        const int maskH = std::round(h * lambda);
        if (maskW == 0 || maskH == 0) {
          inputs = noGrad(in);
          example1Weight = 0.;
        } else if (maskW == w || maskH == h) {
          inputs = noGrad(example1[kImagenetInputIdx]);
          example1Weight = 1.;
        } else {
          const int x = std::rand() % (w - maskW);
          const int y = std::rand() % (h - maskH);

          in(af::seq(x, x + maskW - 1),
             af::seq(y, y + maskH - 1),
             af::span,
             af::span) =
              example1[kImagenetInputIdx](
                  af::seq(x, x + maskW - 1),
                  af::seq(y, y + maskH - 1),
                  af::span,
                  af::span);
          inputs = noGrad(in);
          example1Weight = lambda * lambda;
        }
      }

      // Get the activations from the model.
      auto output = model->forward({inputs}).front();

      // Make a Variable from the target array.
      auto target = noGrad(example[kImagenetTargetIdx]);
      auto target1 = noGrad(example1[kImagenetTargetIdx]);

      // Compute and record the loss.
      auto loss = example1Weight * categoricalCrossEntropy(output, target1) +
          (1 - example1Weight) * categoricalCrossEntropy(output, target);

      trainLossMeter.add(loss.array());
      top5Acc.add(output.array(), target.array());
      top1Acc.add(output.array(), target.array());

      // Backprop, update the weights and then zero the gradients.
      loss.backward();

      if (FLAGS_distributed_enable) {
        reducer->finalize();
      }
      // clamp gradients
      if (FLAGS_train_maxgradnorm > 0) {
        fl::clipGradNorm(model->params(), FLAGS_train_maxgradnorm);
      }
      opt.step();

      // Compute and record the prediction error.
      double trainLoss = trainLossMeter.value()[0];
      if (++idx % 100 == 0) {
        timeMeter.stop();
        fl::ext::syncMeter(trainLossMeter);
        fl::ext::syncMeter(timeMeter);
        fl::ext::syncMeter(top5Acc);
        fl::ext::syncMeter(top1Acc);
        double time = timeMeter.value();
        double samplePerSecond =
            (idx * FLAGS_data_batch_size * worldSize) / time;
        FL_LOG_MASTER(INFO)
            << "Epoch " << epoch << std::setprecision(5) << " Batch: " << idx
            << " Samples per second " << samplePerSecond
            << ": LR: " << opt.getLr() << ": Avg Train Loss: " << trainLoss
            << ": Train Top5 Accuracy( %): " << top5Acc.value()
            << ": Train Top1 Accuracy( %): " << top1Acc.value();
        top5Acc.reset();
        top1Acc.reset();
        trainLossMeter.reset();
        timeMeter.resume();
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
