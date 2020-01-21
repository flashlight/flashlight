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
#include "flashlight/dataset/ImageDataset.h"
#include "flashlight/dataset/PrefetchDataset.h"
#include "flashlight/dataset/ShuffleDataset.h"
#include "flashlight/meter/AverageValueMeter.h"
#include "flashlight/meter/TimeMeter.h"
#include "flashlight/meter/TopKMeter.h"
#include "flashlight/nn/nn.h"
#include "flashlight/optim/optim.h"

using namespace fl;

std::tuple<double, double, double> eval_loop(
    Sequential& model,
    BatchDataset& dataset) {
  AverageValueMeter loss_meter;
  TopKMeter top5_meter(5, true);
  TopKMeter top1_meter(1, true);

  // Place the model in eval mode.
  model.eval();
  for (auto& example : dataset) {
    auto inputs = noGrad(example[ImageDataset::INPUT_IDX]);
    auto output = model(inputs);

    auto target = noGrad(example[ImageDataset::TARGET_IDX]);

    // Compute and record the loss.
    auto loss = categoricalCrossEntropy(output, target);
    loss_meter.add(loss.array().scalar<float>());
    top5_meter.add(output.array(), target.array());
    top1_meter.add(output.array(), target.array());
  }
  // Place the model back into train mode.
  model.train();

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
  const int batchsize = 128;
  const float learning_rate = 0.1f;
  const float momentum = 0.9f;
  const float weight_decay = 0.0001f;
  const int epochs = 100;

  //////////////////////////
  //  Create datasets
  /////////////////////////
  std::unordered_map<std::string, uint32_t> labels =
      ImageDataset::parseLabels(label_path);
  auto train_image_ds = std::make_shared<ImageDataset>(train_list, labels);
  // Shuffle the images before batching to ensure batches are shuffled as well
  auto train_shuffled_ds = std::make_shared<ShuffleDataset>(train_image_ds);
  auto train_prefetch_ds =
      std::make_shared<PrefetchDataset>(train_shuffled_ds, 12, batchsize);
  auto train_ds = BatchDataset(train_prefetch_ds, batchsize);

  // Repeat for val ds
  auto val_image_ds = std::make_shared<ImageDataset>(val_list, labels);
  auto val_shuffled_ds = std::make_shared<ShuffleDataset>(val_image_ds);
  auto val_prefetch_ds =
      std::make_shared<PrefetchDataset>(val_shuffled_ds, 12, batchsize);
  auto val_ds = BatchDataset(val_prefetch_ds, batchsize);

  //////////////////////////
  //  Load model and optimizer
  /////////////////////////
  Sequential model = resnet34();
  SGDOptimizer opt(model.params(), learning_rate, momentum, weight_decay);

  // The main training loop
  for (int e = 0; e < epochs; e++) {
    AverageValueMeter train_loss_meter;
    TimeMeter time_meter;
    TopKMeter top5_meter(5, true);
    TopKMeter top1_meter(1, true);
    train_shuffled_ds->resample();

    // Get an iterator over the data
    time_meter.resume();
    int idx = 0;
    for (auto& example : train_ds) {
      // Make a Variable from the input array.
      auto inputs = noGrad(example[ImageDataset::INPUT_IDX]);

      // Get the activations from the model.
      auto output = model(inputs);

      // Make a Variable from the target array.
      auto target = noGrad(example[ImageDataset::TARGET_IDX]);

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);
      train_loss_meter.add(loss.array());
      top5_meter.add(output.array(), target.array());
      top1_meter.add(output.array(), target.array());

      // Backprop, update the weights and then zero the gradients.
      loss.backward();
      opt.step();
      opt.zeroGrad();

      // Compute and record the prediction error.
      double train_loss = train_loss_meter.value()[0];
      if (++idx % 10 == 0) {
        double time = time_meter.value();
        double sample_per_second = (idx * batchsize) / time;
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
