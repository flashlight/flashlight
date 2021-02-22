/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <iomanip>
#include <iostream>

#include "flashlight/ext/common/DistributedUtils.h"
#include "flashlight/ext/image/fl/models/ViT.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

#include "flashlight/app/imgclass/dataset/Imagenet.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"

#define FL_LOG_MASTER(lvl) FL_LOG_IF(lvl, (fl::getWorldRank() == 0))

int main(int argc, char** argv) {
  fl::init();

  auto a = fl::VisionTransformer::initLinear(1000, 1000).array();
  auto res = fl::ext::afToVector<float>(a);
  // af_print(a);
  // af_print(max(a));
  // af_print(min(a));
  std::ofstream fout(
      "/private/home/qiantong/tmp/fl_trunc_normal.bin", std::ios::binary);
  fout.write((char*)res.data(), res.size() * sizeof(float));

  return 0;

#if 0
  std::vector<float> weights = {1, 3, 4, 5, 2, 6};
  auto arr = af::array(3, 2, weights.data());
  af_print(arr);
  auto network = fl::Linear(fl::Variable(arr, true));
  auto opt = fl::AdamOptimizer(
      network.params(),
      0.001, // FLAGS_train_lr,
      0.9, // FLAGS_train_beta1,
      0.999, // FLAGS_train_beta2,
      1e-8,
      0.05 // FLAGS_train_wd
  );

  std::vector<float> inputs = {1, 5};
  auto input = fl::noGrad(af::array(2, inputs.data()));
  for (int i = 0; i < 100; i++) {
    auto loss = fl::sum(network(input), {0});
    af_print(loss.array());
    loss.backward();
    opt.step();
  }

  fl::ext::initDistributed(0, 1, 8, "");
  af::info();
  if (argc < 3) {
    std::cout
        << "Invalid arguments. Usage : <binary> <batchsize> <precision> <optim_level>"
        << std::endl;
    return 1;
  }
  const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();

  if (std::stoi(argv[2]) == 16) {
    // Only set the optim mode to O1 if it was left empty
    std::cout << "Mixed precision training enabled. Will perform loss scaling."
              << std::endl;
    auto optim_level = std::stoi(argv[3]);
    if (optim_level == 1) {
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
    } else if (optim_level == 2) {
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O2);
    } else if (optim_level == 3) {
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O3);
    } else {
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::DEFAULT);
    }
  }
#endif

  // 1.
  auto network = std::make_shared<fl::ext::image::ViT>(
      12, // FLAGS_model_layers,
      768, // FLAGS_model_hidden_emb_size,
      3072, // FLAGS_model_mlp_size,
      12, // FLAGS_model_heads,
      0.1, // FLAGS_train_dropout,
      0.0, // FLAGS_train_layerdrop,
      1000);

  // auto network = std::make_shared<fl::VisionTransformer>(
  //     768, // hiddenEmbSize_,
  //     64, // hiddenEmbSize_ / nHeads_,
  //     3072, // mlpSize_,
  //     12, // nHeads_,
  //     0, // pDropout,
  //     0.1 // pLayerDrop * (i + 1) / nLayers_
  // );

  FL_LOG_MASTER(fl::INFO) << "[Network] arch - " << network->prettyString();
  FL_LOG_MASTER(fl::INFO) << "[Network] params - "
                          << fl::numTotalParams(network);

  for (auto i : network->modules()) {
    std::cout << i->prettyString() << std::endl;
  }
  return 0;

#if 0
  // 2.
  auto opt = fl::AdamOptimizer(
      network->params(),
      0.1, // FLAGS_train_lr,
      0.9, // FLAGS_train_beta1,
      0.99, // FLAGS_train_beta2,
      1e-8,
      0.1 // FLAGS_train_wd
  );
  auto reducer = std::make_shared<fl::CoalescingReducer>(1.0, true, true);
  fl::distributeModuleGrads(network, reducer);

  // 3.
  const std::string labelPath =
      "/datasets01/imagenet_full_size/061417/labels.txt";
  const std::string trainList = "/datasets01/imagenet_full_size/061417/train";
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  const int randomResizeMax = 480;
  const int randomResizeMin = 256;
  const int randomCropSize = 224;
  const float horizontalFlipProb = 0.5f;
  fl::ext::image::ImageTransform trainTransforms = fl::ext::image::compose(
      {fl::ext::image::randomResizeTransform(randomResizeMin, randomResizeMax),
       fl::ext::image::randomCropTransform(randomCropSize, randomCropSize),
       fl::ext::image::randomAugmentationTransform(0.5, 2),
       fl::ext::image::randomEraseTransform(0.25),
       fl::ext::image::normalizeImage(mean, std),
       fl::ext::image::randomHorizontalFlipTransform(horizontalFlipProb)});

  auto batch_size = std::stoi(argv[1]);
  const int64_t batchSizePerGpu = batch_size;
  const int64_t prefetchThreads = 10;
  const int64_t prefetchSize = batch_size;
  auto labelMap = fl::app::imgclass::getImagenetLabels(labelPath);
  auto trainDataset = std::make_shared<fl::ext::image::DistributedDataset>(
      fl::app::imgclass::imagenetDataset(
          trainList, labelMap, {trainTransforms}),
      worldRank,
      worldSize,
      batchSizePerGpu,
      3, // FLAGS_train_n_repeatedaug,
      prefetchThreads,
      prefetchSize,
      fl::BatchDatasetPolicy::SKIP_LAST);
  FL_LOG_MASTER(fl::INFO) << "[trainDataset size] " << trainDataset->size();

  // 4.
  auto input = fl::Variable(af::randu(224, 224, 3, batch_size), false);
  // auto input = fl::Variable(af::randu(768, 197, batch_size), false);
  input.zeroGrad();
  auto target = fl::Variable(af::randu(1000, batch_size), false);
  if (std::stoi(argv[2]) == 16) {
    // input = input.as(af::dtype::f16);
    target = target.as(af::dtype::f16);
  }

  // RUN !!!
  int warmup = 15, n = 100;
  network->train();
  for (int i = 0; i < warmup; i++) {
    network->zeroGrad();
    opt.zeroGrad();

    auto output = network->forward({input}).front();
    // output = fl::mean(fl::sum(fl::negate(target * output), {0}), {1});
    output.backward();

    if (fl::getWorldSize() > 1) {
      reducer->finalize();
    }
    // fl::clipGradNorm(network->params(), 0.1);
    opt.step();
  }

  double smp_time = 0., fwd_time = 0., bwd_time = 0., optim_time = 0.,
         total_time = 0.;
  auto start = af::timer::start();
  for (int i = 0; i < n; i++) {
    network->zeroGrad();
    opt.zeroGrad();

    // sample
    auto start1 = af::timer::start();
    auto sample = trainDataset->get(i);
    auto rawInput = sample[fl::app::imgclass::kImagenetInputIdx];
    input = fl::Variable(rawInput, false);

    // af::array inputtmp = af::randu(224, 224, 3);
    // for (int i = 1; i < batch_size; i++) {
    //   auto tmp = af::randu(224, 224, 3);
    //   tmp = trainTransforms(tmp);
    //   inputtmp = af::join(3, inputtmp, tmp);
    // }
    // auto input = fl::Variable(inputtmp, false);
    af::sync();
    smp_time += af::timer::stop(start1);

    // fwd
    start1 = af::timer::start();
    auto output = network->forward({input}).front();
    // output = fl::mean(fl::sum(fl::negate(target * output), {0}), {1});
    af::sync();
    fwd_time += af::timer::stop(start1);

    // bwd
    start1 = af::timer::start();
    output.backward();
    if (fl::getWorldSize() > 1) {
      reducer->finalize();
    }
    af::sync();
    bwd_time += af::timer::stop(start1);

    // optim
    start1 = af::timer::start();
    // fl::clipGradNorm(network->params(), 0.1);
    opt.step();
    af::sync();
    optim_time += af::timer::stop(start1);
  }
  total_time += af::timer::stop(start);

  std::cout << "batch time: " << total_time * 1000 / n << "ms" << std::endl;
  std::cout << "smp time: " << smp_time * 1000 / n << "ms" << std::endl;
  std::cout << "fwd time: " << fwd_time * 1000 / n << "ms" << std::endl;
  std::cout << "bwd time: " << bwd_time * 1000 / n << "ms" << std::endl;
  std::cout << "optim time: " << optim_time * 1000 / n << "ms" << std::endl;

  std::cout << "Throughput/GPU: " << batch_size * n / total_time << " images/s"
            << std::endl;
#endif
  return 0;
}