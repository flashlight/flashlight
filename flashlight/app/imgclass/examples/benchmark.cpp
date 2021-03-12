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

#include <unistd.h>

#define FL_LOG_MASTER(lvl) FL_LOG_IF(lvl, (fl::getWorldRank() == 0))

const float pi = std::acos(-1);

float randomFloat(float a, float b) {
  float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  return a + (b - a) * r;
}

af::array resize(const af::array& in, const int resize) {
  return af::resize(in, resize, resize, AF_INTERP_BILINEAR);
}

af::array
crop(const af::array& in, const int x, const int y, const int w, const int h) {
  return in(af::seq(x, x + w - 1), af::seq(y, y + h - 1), af::span, af::span);
}

af::array resizeSmallest(const af::array& in, const int resize) {
  const int w = in.dims(0);
  const int h = in.dims(1);
  int th, tw;
  if (h > w) {
    th = (resize * h) / w;
    tw = resize;
  } else {
    th = resize;
    tw = (resize * w) / h;
  }
  return af::resize(in, tw, th, AF_INTERP_BILINEAR);
}

af::array randomResizeCropTransform(const af::array& in) {
  const int size = 224;
  const float scaleLow = 0.08;
  const float scaleHigh = 1.0;
  const float ratioLow = 3. / 4.;
  const float ratioHigh = 4. / 3.;

  const int w = in.dims(0);
  const int h = in.dims(1);
  const float area = w * h;
  for (int i = 0; i < 10; i++) {
    const float scale = randomFloat(scaleLow, scaleHigh);
    const float logRatio = randomFloat(std::log(ratioLow), std::log(ratioHigh));
    ;
    const float targetArea = scale * area;
    const float targetRatio = std::exp(logRatio);
    const int tw = std::round(std::sqrt(targetArea * targetRatio));
    const int th = std::round(std::sqrt(targetArea / targetRatio));
    if (0 < tw && tw <= w && 0 < th && th <= h) {
      const int x = std::rand() % (w - tw + 1);
      const int y = std::rand() % (h - th + 1);
      return resize(crop(in, x, y, tw, th), size);
    }
  }
  return in;
}

af::array randomResizeCropTransformOld(const af::array& in) {
  const int low = 256;
  const int high = 480;
  const float scale =
      static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  const int resize = low + (high - low) * scale;
  auto res = resizeSmallest(in, resize);

  const uint64_t w = res.dims(0);
  const uint64_t h = res.dims(1);
  const int size = 224;
  const int x = std::rand() % (w - size + 1);
  const int y = std::rand() % (h - size + 1);
  return crop(res, x, y, size, size);
}

af::array randomerase(const af::array& in) {
  const float areaRatioMin = 0.02;
  const float areaRatioMax = 1 / 3.;
  const float edgeRatioMin = 0.3;
  const float edgeRatioMax = 10 / 3.;

  const int w = in.dims(0);
  const int h = in.dims(1);
  const int c = in.dims(2);

  af::array out = in;
  for (int i = 0; i < 10; i++) {
    float s = w * h * randomFloat(areaRatioMin, areaRatioMax);
    float r =
        std::exp(randomFloat(std::log(edgeRatioMin), std::log(edgeRatioMax)));
    int maskW = std::round(std::sqrt(s * r));
    int maskH = std::round(std::sqrt(s / r));
    if (maskW >= w || maskH >= h) {
      continue;
    }

    const int x = std::rand() % (w - maskW);
    const int y = std::rand() % (h - maskH);
    af::array fillValue =
        (af::randu(af::dim4(maskW, maskH, c)) * 255).as(in.type());
    // std::cout << x << " " << y << " " << maskW << " " << maskH << std::endl;
    // af::array fillValue = af::randn(af::dim4(maskW, maskH, c), in.type());

    out(af::seq(x, x + maskW - 1),
        af::seq(y, y + maskH - 1),
        af::span,
        af::span) = fillValue;
    break;
  }
  out.eval();
  // std::cout << af::allTrue<bool>(out == in) << std::endl;
  return out;
};

af::array flip(const af::array& in) {
  const uint64_t w = in.dims(0);
  auto out = in(af::seq(w - 1, 0, -1), af::span, af::span, af::span);
  // const uint64_t w = in.dims(1);
  // out = out(af::span, af::seq(w - 1, 0, -1), af::span, af::span);

  return out;
}

int main(int argc, char** argv) {
  fl::init();

#if 0
  for (int j = 0; j < 10; j++) {
    std::vector<int> selectOp(15);
    std::iota(selectOp.begin(), selectOp.end(), 0);
    auto nOp = 15, nSelected = 0;
    // custom implementation of shuffle - https://stackoverflow.com/a/51931164
    for (auto i = nOp; i >= 1; --i) {
      std::swap(selectOp[i - 1], selectOp[std::rand() % nOp]);
    }
    for (auto i = 0; i < nOp; i++) {
      std::cout << selectOp[i] << ",";
    }
    std::cout << std::endl;
  }
  return 0;

  auto data = af::randn(af::dim4(1e4), f32);
  af_print(af::min(data, 0));
  af_print(af::max(data, 0));
  af_print(af::mean(data, 0));
  af_print(af::stdev(data, 0));

  auto img = af::randu(224, 224, 3);
  for (int i = 0; i < 1e7; i++) {
    randomerase(img);
  }
  return 0;

  auto worldRank = std::stoi(argv[1]);
  // fl::ext::initDistributed(worldRank, 2, 8, "/tmp/rndv");
  // af::info();
  // // const int worldRank = fl::getWorldRank();
  const int worldSize = fl::getWorldSize();
  // std::srand(worldRank * 4399);

  // auto deviceId = af::getDevice();
  // std::unique_ptr<fl::ThreadPool> threadPool =
  // std::make_unique<fl::ThreadPool>(
  //     5, [deviceId](int /* threadId */) { af::setDevice(deviceId); });

  // for (int i = 0; i < 10; i++) {
  //   threadPool->enqueue([i]() {
  //     usleep((10 - i) * 2e5);
  //     std::cout << std::rand() << std::endl;
  //     // std::cout << "wtf: " << std::rand() << " / " << RAND_MAX <<
  //     std::endl; return;
  //   });
  // }
  // return 0;
  const std::string labelPath =
      "/datasets01/imagenet_full_size/061417/labels.txt";
  const std::string trainList = "/datasets01/imagenet_full_size/061417/train";
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  const int randomResizeMax = 480;
  const int randomResizeMin = 256;
  const int randomCropSize = 224;
  const float horizontalFlipProb = 0.5f;
  fl::ext::image::ImageTransform trainTransforms = fl::ext::image::compose({
      fl::ext::image::randomResizeTransform(randomResizeMin, randomResizeMax),
      fl::ext::image::randomCropTransform(randomCropSize, randomCropSize),
      fl::ext::image::randomHorizontalFlipTransform(horizontalFlipProb),
      //  fl::ext::image::randomAugmentationTransform(0.5, 2),
      fl::ext::image::randomAugmentationTransform(1, 1),
      fl::ext::image::normalizeImage(mean, std),
      fl::ext::image::randomEraseTransform(0.25)
      // end
  });

  auto batch_size = 3;
  const int64_t batchSizePerGpu = batch_size;
  const int64_t prefetchThreads = 3;
  const int64_t prefetchSize = batch_size * 10;
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

  for (int i = 0; i < 10; i++) {
    // if (worldRank == 1) {
    //   usleep(2e6);
    // }
    auto sample = trainDataset->get(i);
    fl::allReduce(sample[0]);
    usleep(1e6);
    fl::barrier();
    std::cout << "---" << worldRank << ": " << i << "\n";
  }

  return 0;
#endif

  auto img = fl::ext::image::loadJpeg(
      "/datasets01/imagenet_full_size/061417/test/ILSVRC2012_test_00036355.JPEG");
  std::cout << img.type() << std::endl;

  std::string root = "/private/home/qiantong/tmp/fl_trans/";
  for (int i = 0; i < 10; i++) {
    fl::ext::image::saveJpeg(
        root + "re" + std::to_string(i) + ".JPEG", randomerase(img));
  }

  for (int i = 0; i < 10; i++) {
    fl::ext::image::saveJpeg(
        root + "rci" + std::to_string(i) + ".JPEG",
        randomResizeCropTransform(img));
  }

  for (int i = 0; i < 10; i++) {
    fl::ext::image::saveJpeg(
        root + "rci_old" + std::to_string(i) + ".JPEG",
        randomResizeCropTransformOld(img));
  }

  fl::ext::image::saveJpeg(root + "flip.JPEG", flip(img));
  const int w = img.dims(0);
  const int h = img.dims(1);
  const int c = img.dims(2);

  const std::vector<float> mean = {0.485 * 256, 0.456 * 256, 0.406 * 256};
  auto background = af::array(1, 1, 3, 1, mean.data());
  background = af::tile(background, w, h);
  std::cout << background.dims() << std::endl;

  {
    // rotate
    auto res = img.as(f32) + 0.1;
    float theta = pi / 6;
    res = af::rotate(res, theta);

    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3).as(f32);
    res = mask * background + (1 - mask) * res;

    fl::ext::image::saveJpeg(root + "rotate.JPEG", res.as(img.type()));
  }
  {
    // rotate
    auto res = img.as(f32) + 0.1;
    float theta = -pi / 6;
    res = af::rotate(res, theta);

    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3).as(f32);
    res = mask * background + (1 - mask) * res;

    fl::ext::image::saveJpeg(root + "rotate1.JPEG", res.as(img.type()));
  }
  {
    // skew
    auto res = img.as(f32) + 0.1;
    float theta = pi / 6;
    res = af::skew(res, theta, 0);

    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3).as(f32);
    res = mask * background + (1 - mask) * res;

    fl::ext::image::saveJpeg(root + "skew1.JPEG", res.as(img.type()));
  }
  {
    // skew
    auto res = img.as(f32) + 0.1;
    float theta = pi / 6;
    res = af::skew(res, 0, theta);

    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3).as(f32);
    res = mask * background + (1 - mask) * res;

    fl::ext::image::saveJpeg(root + "skew2.JPEG", res.as(img.type()));
  }
  {
    // translate
    auto res = img.as(f32) + 0.1;
    int shift = 100;
    res = af::translate(res, shift, 0);

    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3).as(f32);
    res = mask * background + (1 - mask) * res;

    fl::ext::image::saveJpeg(root + "translate1.JPEG", res.as(img.type()));
  }
  {
    // translate
    auto res = img.as(f32) + 0.1;
    int shift = 100;
    res = af::translate(res, 0, shift);

    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3).as(f32);
    res = mask * background + (1 - mask) * res;

    fl::ext::image::saveJpeg(root + "translate2.JPEG", res.as(img.type()));
  }
  {
    // color
    auto res = img;
    float enhance = 1.8;
    auto meanPic = af::mean(res, 2);
    meanPic = af::tile(meanPic, af::dim4(1, 1, c));
    res = af::clamp(meanPic + enhance * (res - meanPic), 0., 255.);
    fl::ext::image::saveJpeg(root + "color.JPEG", res);
  }
  {
    // color
    auto res = img;
    float enhance = 0.2;
    auto meanPic = af::mean(res, 2);
    meanPic = af::tile(meanPic, af::dim4(1, 1, c));
    res = af::clamp(meanPic + enhance * (res - meanPic), 0., 255.);
    fl::ext::image::saveJpeg(root + "color_neg.JPEG", res);
  }
  {
    // auto contrast
    auto res = img;
    float enhance = 1.5;
    auto minPic = res;
    auto maxPic = res;
    for (int j = 0; j < 3; j++) {
      minPic = af::min(minPic, j);
      maxPic = af::max(maxPic, j);
    }
    minPic = af::tile(minPic, af::dim4(w, h, c));
    maxPic = af::tile(maxPic, af::dim4(w, h, c));
    auto scale = 256. / (maxPic - minPic + 1);
    res = af::clamp(scale * (res - minPic), 0., 255.);
    fl::ext::image::saveJpeg(root + "auto_contrast.JPEG", res);
  }
  {
    // contrast
    auto res = img;
    float enhance = 1.8;
    auto meanPic = res;
    for (int j = 0; j < 3; j++) {
      meanPic = af::mean(meanPic, j);
    }
    meanPic = af::tile(meanPic, af::dim4(w, h, c));
    res = af::clamp(meanPic + enhance * (res - meanPic), 0., 255.);
    fl::ext::image::saveJpeg(root + "contrast.JPEG", res);
  }
  {
    // contrast
    auto res = img;
    float enhance = 0.2;
    auto meanPic = res;
    for (int j = 0; j < 3; j++) {
      meanPic = af::mean(meanPic, j);
    }
    meanPic = af::tile(meanPic, af::dim4(w, h, c));
    res = af::clamp(meanPic + enhance * (res - meanPic), 0., 255.);
    fl::ext::image::saveJpeg(root + "contrast_neg.JPEG", res);
  }
  {
    // brightness
    auto res = img;
    float enhance = 1.8;
    res = af::clamp(res * enhance, 0., 255.);
    fl::ext::image::saveJpeg(root + "brightness.JPEG", res);
  }
  {
    // brightness
    auto res = img;
    float enhance = 0.2;
    res = af::clamp(res * enhance, 0., 255.);
    fl::ext::image::saveJpeg(root + "brightness_neg.JPEG", res);
  }
  {
    // invert
    auto res = img;
    res = 255 - res;
    fl::ext::image::saveJpeg(root + "invert.JPEG", res);
  }
  {
    // solarize
    auto res = img;
    auto mask = (res < 25.).as(f32);
    res = mask * res + (1 - mask) * (255 - res);
    fl::ext::image::saveJpeg(root + "solarize.JPEG", res);
  }
  {
    // solarize
    auto res = img;
    auto mask = (res < 128.).as(f32);
    res = mask * (res + 100) + (1 - mask) * res;
    fl::ext::image::saveJpeg(root + "solarize_add.JPEG", res);
  }
  {
    // equalize
    auto res = img;
    res = flat(res);
    auto hist = af::histogram(res, 256);
    res = af::histEqual(res, hist);
    res = moddims(res, img.dims());
    fl::ext::image::saveJpeg(root + "equalize.JPEG", res);
  }
  {
    // posterize
    auto res = img;
    int mask = ~127; // 0x1f
    res = res.as(s32) & mask;
    fl::ext::image::saveJpeg(root + "posterize.JPEG", res);
  }
  {
    // sharpness
    auto res = img;
    float enhance = 1;

    auto meanPic = af::mean(res, 2);
    auto blurKernel = af::gaussianKernel(7, 7);
    auto blur = af::convolve(meanPic, blurKernel);
    auto diff = af::tile(meanPic - blur, af::dim4(1, 1, c));

    res = af::clamp(res + enhance * diff, 0., 255.);
    fl::ext::image::saveJpeg(root + "sharpness.JPEG", res);
  }
  {
    // sharpness
    auto res = img;
    float enhance = -1;

    auto meanPic = af::mean(res, 2);
    auto blurKernel = af::gaussianKernel(7, 7);
    auto blur = af::convolve(meanPic, blurKernel);
    auto diff = af::tile(meanPic - blur, af::dim4(1, 1, c));

    res = af::clamp(res + enhance * diff, 0., 255.);
    fl::ext::image::saveJpeg(root + "sharpness_neg.JPEG", res);
  }

  return 0;

#if 0
  auto a = fl::VisionTransformer::initLinear(1000, 1000).array();
  auto res = fl::ext::afToVector<float>(a);
  // af_print(a);
  // af_print(max(a));
  // af_print(min(a));
  std::ofstream fout(
      "/private/home/qiantong/tmp/fl_trunc_normal.bin", std::ios::binary);
  fout.write((char*)res.data(), res.size() * sizeof(float));

  return 0;

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