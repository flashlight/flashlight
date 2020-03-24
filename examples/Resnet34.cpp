/**
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <exception>
#include <iomanip>

#include <cudnn.h>

#include "flashlight/contrib/modules/Resnet.h"
#include "flashlight/dataset/BatchDataset.h"
#include "flashlight/dataset/Dataset.h"
#include "flashlight/dataset/ImagenetUtils.h"
#include "flashlight/dataset/PrefetchDataset.h"
#include "flashlight/dataset/ShuffleDataset.h"
#include "flashlight/meter/AverageValueMeter.h"
#include "flashlight/meter/TimeMeter.h"
#include "flashlight/meter/TopKMeter.h"
#include "flashlight/nn/nn.h"
#include "flashlight/optim/optim.h"

#define DISTRIBUTED 1

using namespace fl;

class DistributedDataset : public Dataset {
 public:
  DistributedDataset(
      std::shared_ptr<Dataset> base,
      int64_t world_rank,
      int64_t world_size,
      int64_t batch_size,
      int64_t num_threads,
      int64_t prefetch_size) {
    shuffle_ = std::make_shared<ShuffleDataset>(base);
    auto permfn = [world_size, world_rank](int64_t idx) {
      return (idx * world_size) + world_rank;
    };
    ds_ = std::make_shared<ResampleDataset>(
	shuffle_, permfn, shuffle_->size() / world_size);
    ds_ = std::make_shared<PrefetchDataset>(ds_, num_threads, prefetch_size);
    ds_ = std::make_shared<BatchDataset>(ds_, batch_size);
  }

  std::vector<af::array> get(const int64_t idx) const override {
    checkIndexBounds(idx);
    return ds_->get(idx);
  }

  void resample() {
    shuffle_->resample();
  }

  int64_t size() const override {
    return ds_->size();
  }

 private:
  std::shared_ptr<Dataset> ds_;
  std::shared_ptr<ShuffleDataset> shuffle_;
};

std::tuple<double, double, double> eval_loop(
    std::shared_ptr<Sequential> model,
    Dataset& dataset) {
  AverageValueMeter loss_meter;
  TopKMeter top5_meter(5, true);
  TopKMeter top1_meter(1, true);

  // Place the model in eval mode.
  model->eval();
  int idx = 0;
  for (auto& example : dataset) {
    auto inputs = noGrad(example[ImageDataset::INPUT_IDX]);
    auto output = model->forward(inputs);

    auto target = noGrad(example[ImageDataset::TARGET_IDX]);

    // Compute and record the loss.
    auto loss = categoricalCrossEntropy(output, target);
    loss_meter.add(loss.array().scalar<float>());
    top5_meter.add(output.array(), target.array());
    top1_meter.add(output.array(), target.array());
    idx++;
  }
  // Place the model back into train mode.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  model->train();

  double top1_error = top1_meter.value();
  double top5_error = top5_meter.value();
  double loss = loss_meter.value()[0];
  return std::make_tuple(loss, top5_error, top1_error);
};


int main(int argc, const char** argv) {
  if (argc < 2) {
    std::cout << "Must specify imagenet data location" << std::endl;
    return -1;
  }
  const std::string imagenet_base = argv[1];

  int world_rank = argc > 2 ? atoi(argv[2]) : 0;
  int world_size = argc > 3 ? atoi(argv[3]) : 1;
  af::setDevice(world_rank);
  if (world_size > 1 && !DISTRIBUTED) {
    std::cout << "Not built for distributed!" << std::endl;
    return -1;
  }

  const std::string label_path = imagenet_base + "labels.txt";
  const std::string train_list = imagenet_base + "train";
  const std::string val_list = imagenet_base + "val";

  /////////////////////////
  // Hyperparaters
  ////////////////////////
  const int batch_size = 256;
  const float learning_rate = 0.1f;
  const float momentum = 0.9f;
  const float weight_decay = 0.0001f;
  const int epochs = 300;

  /////////////////////////
  // Setup distributed training
  ////////////////////////
  af::info();
#if DISTRIBUTED
  fl::distributedInit(
	fl::DistributedInit::FILE_SYSTEM,
	world_rank,
	world_size,
	{{fl::DistributedConstants::kMaxDevicePerNode,
	  std::to_string(8)},
	 {fl::DistributedConstants::kFilePath, "/checkpoint/padentomasello/tmp/" + std::to_string(world_size)}});

  std::cout << "WorldRank" << world_rank << "world_size " << world_size << std::endl;
  af::setSeed(world_size);

  auto reducer = std::make_shared<fl::CoalescingReducer>(
      1.0 / world_size,
      true,
      true);
#endif




  //////////////////////////
  //  Create datasets
  /////////////////////////
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  auto labels = imagenetLabels(label_path);
  std::vector<Dataset::TransformFunction> train_transforms = {
      // randomly resize shortest side of image between 256 to 480 for scale
      // invariance
      ImageDataset::randomResizeTransform(256, 480),
      ImageDataset::randomCropTransform(224, 224),
      ImageDataset::normalizeImage(mean, std),
      // Randomly flip image with probability of 0.5
      ImageDataset::horizontalFlipTransform(0.5)
  };
  std::vector<Dataset::TransformFunction> val_transforms = {
      // Resize shortest side to 256, then take a center crop
      ImageDataset::resizeTransform(256),
      ImageDataset::centerCrop(224),
      ImageDataset::normalizeImage(mean, std)
  };
  const uint64_t miniBatchSize = batch_size / world_size;
  const int64_t prefetch_threads = 10;
  const int64_t prefetch_size = miniBatchSize * 2;
  auto test = std::make_shared<ImageDataset>(
          imagenetDataset(train_list, labels, train_transforms));
  auto train_ds = DistributedDataset(
      test,
      world_rank,
      world_size,
      miniBatchSize,
      prefetch_threads,
      prefetch_size);

  auto val_ds = DistributedDataset(
      std::make_shared<ImageDataset>(
          imagenetDataset(val_list, labels, val_transforms)),
      world_rank,
      world_size,
      miniBatchSize,
      prefetch_threads,
      prefetch_size);


  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  auto model = std::make_shared<Sequential>(resnet34());
  // synchronize parameters of tje model so that the parameters in each process
  // is the same

  SGDOptimizer opt(model->params(), learning_rate, momentum, weight_decay);

  auto lrScheduler = [&opt, &learning_rate](int epoch) {
    // Adjust learning rate every 30 epoch
    if (epoch > 0 && epoch % 30 == 0) {
      const float newLr = opt.getLr() * 0.1;
      std::cout << "Setting learning rate to: " << newLr << std::endl;
      opt.setLr(newLr);
    }
  };

  // Small utility functions to load and save models
  std::string checkpointPrefix = "/private/home/padentomasello/code/flashlight/build/model-max";
  auto saveModel = [world_rank, &model, &checkpointPrefix](int epoch) {
    if(world_rank == 0) {
      std::string modelPath = checkpointPrefix + std::to_string(epoch);
      std::cout <<  "Saving model to file: " << modelPath << std::endl;
      fl::save(modelPath, model);
    }
  };

  auto loadModel = [&model, &checkpointPrefix](int epoch) {
      std::string modelPath = checkpointPrefix + std::to_string(epoch);
      std::cout <<  "Loading model from file: " << modelPath << std::endl;
      fl::load(modelPath, model);
  };
  int checkpointEpoch = -1;
  if (checkpointEpoch >= 0) {
    loadModel(checkpointEpoch);
  }

#if DISTRIBUTED
  fl::allReduceParameters(model);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(model, reducer);
#endif

#if 0
  for (int i = 0; i < epochs; i++) {
    int idx = 0;
    for (auto& example : train_ds) {
      auto test = example[ImageDataset::INPUT_IDX].scalar<float>();
      if (test == 0.0) {
        std::cout << "Impossible " << std::endl;
      }
      if (idx % 10) {
        std::cout << "Epoch" << i << "Idx " << idx << std::endl;
      }
      idx++;
    }
}

#else
  // The main training loop
  TimeMeter time_meter;
  TopKMeter top5_meter(5, true);
  TopKMeter top1_meter(1, true);
  AverageValueMeter train_loss_meter;
  for (int e = (checkpointEpoch + 1); e < epochs; e++) {
    train_ds.resample();
    lrScheduler(e);

    // Get an iterator over the data
    time_meter.resume();
    int idx = 0;
    for (auto& example : train_ds) {
      opt.zeroGrad();
      // Make a Variable from the input array.
      auto inputs = noGrad(example[ImageDataset::INPUT_IDX]);

      // Get the activations from the model.
      auto output = model->forward(inputs);

      // Make a Variable from the target array.
      auto target = noGrad(example[ImageDataset::TARGET_IDX]);

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);

      train_loss_meter.add(loss.array());
      top5_meter.add(output.array(), target.array());
      top1_meter.add(output.array(), target.array());

      // Backprop, update the weights and then zero the gradients.
      loss.backward();

#if DISTRIBUTED
      reducer->finalize();
#endif
      opt.step();

      // Compute and record the prediction error.
      double train_loss = train_loss_meter.value()[0];
      if (++idx % 50 == 0) {
        double time = time_meter.value();
        double sample_per_second = (idx * batch_size) / time;
        std::cout << "Epoch " << e << std::setprecision(5) << " Batch: " << idx
                  << " Samples per second " << sample_per_second
                  << ": Avg Train Loss: " << train_loss
                  << ": Train Top5 Error( %): " << top5_meter.value()
                  << ": Train Top1 Error( %): " << top1_meter.value()
                  << std::endl;
        top5_meter.reset();
        top1_meter.reset();
        train_loss_meter.reset();
      }
    }
    time_meter.reset();
    time_meter.stop();

    // Evaluate on the dev set.
    double val_loss, val_top1_err, val_top5_err;
    std::tie(val_loss, val_top5_err, val_top1_err) = eval_loop(model, val_ds);

    std::cout << "Epoch " << e << std::setprecision(5)
              << " Validation Loss: " << val_loss
              << " Validation Top5 Error (%): " << val_top5_err
              << " Validation Top1 Error (%): " << val_top1_err << std::endl;
    saveModel(e);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
#endif
}
