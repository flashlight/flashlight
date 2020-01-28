/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <exception>
#include <iomanip>

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
  for (auto& example : dataset) {
    auto inputs = noGrad(example[ImageDataset::INPUT_IDX]);
    auto output = model->forward(inputs);

    auto target = noGrad(example[ImageDataset::TARGET_IDX]);

    // Compute and record the loss.
    auto loss = categoricalCrossEntropy(output, target);
    loss_meter.add(loss.array().scalar<float>());
    top5_meter.add(output.array(), target.array());
    top1_meter.add(output.array(), target.array());
  }
  // Place the model back into train mode.
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

  int device = argc > 2 ? atoi(argv[2]) : 0;
  af::setDevice(device);

  const std::string label_path = imagenet_base + "labels.txt";
  const std::string train_list = imagenet_base + "train";
  const std::string val_list = imagenet_base + "val";

  /////////////////////////
  // Hyperparaters
  ////////////////////////
  const int batch_size = 128;
  const float learning_rate = 0.1f;
  const float momentum = 0.9f;
  const float weight_decay = 0.0001f;
  const int epochs = 100;

  /////////////////////////
  // Setup distributed training
  ////////////////////////
  af::info();
  fl::distributedInit(
        fl::DistributedInit::FILE_SYSTEM,
        0,
        1,
        {{fl::DistributedConstants::kMaxDevicePerNode,
          std::to_string(8)},
         {fl::DistributedConstants::kFilePath, "/tmp/"}});
  auto world_size = fl::getWorldSize();
  auto world_rank = fl::getWorldRank();
  std::cout << "WorldRank" << world_rank << "world_size " << world_size << std::endl;
  af::setSeed(world_rank);

  auto reducer = std::make_shared<fl::CoalescingReducer>(
      /*scale=*/1.0 / world_size,
      /*async=*/true,
      /*contiguous=*/true);


  //////////////////////////
  //  Create datasets
  /////////////////////////
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  auto labels = imagenetLabels(label_path);
  std::vector<Dataset::TransformFunction> train_transforms = {
      ImageDataset::randomCropTransform(224, 224),
      // Some images are smaller than 224, in which case randomCrop will return
      // the entire image and we still need to resize.
      ImageDataset::resizeTransform(224),
      ImageDataset::normalizeImage(mean, std),
      ImageDataset::horizontalFlipTransform()
  };
  std::vector<Dataset::TransformFunction> val_transforms = {
      ImageDataset::resizeTransform(224),
      ImageDataset::normalizeImage(mean, std)
  };
  const int64_t prefetch_threads = 12;
  const int64_t prefetch_size = batch_size;
  auto train_ds = DistributedDataset(
      std::make_shared<ImageDataset>(
          imagenetDataset(train_list, labels, train_transforms)),
      world_rank,
      world_size,
      batch_size,
      prefetch_threads,
      prefetch_size);

  auto val_ds = DistributedDataset(
      std::make_shared<ImageDataset>(
          imagenetDataset(val_list, labels, val_transforms)),
      world_rank,
      world_size,
      batch_size,
      prefetch_threads,
      prefetch_size);

  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  auto model = std::make_shared<Sequential>(resnet34());
  // synchronize parameters of the model so that the parameters in each process
  // is the same
  fl::allReduceParameters(model);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(model, reducer);

  SGDOptimizer opt(model->params(), learning_rate, momentum, weight_decay);

  // The main training loop
  for (int e = 0; e < epochs; e++) {
    train_ds.resample();
    AverageValueMeter train_loss_meter;
    TimeMeter time_meter;
    TopKMeter top5_meter(5, true);
    TopKMeter top1_meter(1, true);
    //train_shuffled_ds->resample();

    // Get an iterator over the data
    time_meter.resume();
    int idx = 0;
    for (auto& example : train_ds) {
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
      reducer->finalize();
      opt.step();
      opt.zeroGrad();

      // Compute and record the prediction error.
      double train_loss = train_loss_meter.value()[0];
      if (++idx % 10 == 0) {
        double time = time_meter.value();
        double sample_per_second = (idx * batch_size * world_size) / time;
        std::cout << "Epoch " << e << std::setprecision(3) << " Batch: " << idx
                  << " Samples per second " << sample_per_second
                  << ": Avg Train Loss: " << train_loss
                  << ": Train Top5 Error( %): " << top5_meter.value()
                  << ": Train Top1 Error( %): " << top1_meter.value()
                  << std::endl;
        top5_meter.reset();
        top1_meter.reset();
        train_loss_meter.reset();
      }
      time_meter.resume();
    }
    time_meter.stop();

    // Evaluate on the dev set.
    double val_loss, val_top1_err, val_top5_err;
    std::tie(val_loss, val_top5_err, val_top1_err) = eval_loop(model, val_ds);

    std::cout << "Epoch " << e << std::setprecision(3)
              << " Validation Loss: " << val_loss
              << " Validation Top5 Error (%): " << val_top5_err
              << " Validation Top1 Error (%): " << val_top1_err << std::endl;
  }
}
