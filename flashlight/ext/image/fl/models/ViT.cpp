/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/fl/models/ViT.h"

#include <fstream>

namespace fl {
namespace ext {
namespace image {

ViT::ViT(
    const int nLayers,
    const int hiddenEmbSize,
    const int mlpSize,
    const int nHeads,
    const float pDropout,
    const float pLayerDrop,
    const int nClasses,
    const bool usePosEmb,
    const bool useAugPosEmb,
    const int imageSize,
    const int patchSize)
    : nLayers_(nLayers),
      hiddenEmbSize_(hiddenEmbSize),
      mlpSize_(mlpSize),
      nHeads_(nHeads),
      pDropout_(pDropout),
      nClasses_(nClasses),
      usePosEmb_(usePosEmb),
      useAugPosEmb_(useAugPosEmb),
      patchEmbedding_(std::make_shared<Conv2D>(
          3,
          hiddenEmbSize_,
          patchSize,
          patchSize,
          patchSize,
          patchSize)) {
  // Class token
  params_.emplace_back(fl::truncNormal(af::dim4(hiddenEmbSize_, 1), 0.02));

  // Positional embedding
  if (usePosEmb_) {
    int nPatches = imageSize / patchSize;
    params_.emplace_back(fl::truncNormal(
        af::dim4(hiddenEmbSize_, nPatches * nPatches + 1), 0.02));
  } else if (useAugPosEmb_) {
    augPosEmb_ =
        std::make_shared<SinusoidalPositionEmbedding2D>(hiddenEmbSize_, 1.);
    add(augPosEmb_);
  }

  // Modules
  add(patchEmbedding_);

  for (int i = 0; i < nLayers_; ++i) {
    transformers_.emplace_back(std::make_shared<VisionTransformer>(
        hiddenEmbSize_,
        hiddenEmbSize_ / nHeads_,
        mlpSize_,
        nHeads_,
        pDropout,
        pLayerDrop * i / (nLayers_ - 1)));
    add(transformers_.back());
  }

  linearOut_ = std::make_shared<Linear>(
      fl::truncNormal(af::dim4(nClasses, hiddenEmbSize_), 0.02),
      fl::constant(0., nClasses, 1, af::dtype::f32));
  add(linearOut_);

  ln_ = std::make_shared<LayerNorm>(
      std::vector<int>({0}), 1e-6, true, hiddenEmbSize_);
  add(ln_);
}

std::vector<fl::Variable> ViT::forward(
    const std::vector<fl::Variable>& inputs) {
  return forward(inputs, false);
}

void ViT::resetPosEmb(const int newImgSize) {
  if (usePosEmb_) {
    std::cerr << "Reset posemb";
    if (params_[1].dims(1) - 1 != newImgSize * newImgSize) {
      std::cerr << "Reset posemb: do interpolation from "
                << params_[1].dims(1) - 1 << " to " << newImgSize * newImgSize
                << std::endl;
      af::array posEmbArr = params_[1].array();
      af::array clsEmb = posEmbArr.col(0);
      af::array posEmbAll = posEmbArr.cols(1, posEmbArr.dims(1) - 1);

      int imageDim = std::sqrt(posEmbAll.dims(1));
      int C = posEmbAll.dims(0);
      if (imageDim * imageDim != posEmbAll.dims(1)) {
        std::cerr << "ERROR in up/down-sampling";
      }
      // {
      //   int newW = 14;
      //   int oldW = 10;
      //   auto emb = af::randu(af::dim4(768, oldW * oldW));
      //   af::array newEmbTmp = af::transpose(emb);
      //   newEmbTmp = af::moddims(newEmbTmp, oldW, oldW, 768);
      //   auto newEmb = af::resize(newEmbTmp, newW, newW, AF_INTERP_BICUBIC);

      //   for (int i = 0; i < 10; i++) {
      //     af::array tmp1 = af::transpose(emb);
      //     tmp1 = af::moddims(tmp1, oldW, oldW, 768);
      //     auto tmp = af::resize(tmp1, newW, newW, AF_INTERP_BICUBIC);
      //     std::cout << "CHECK" << af::allTrue<bool>(newEmb == tmp) << std::endl;
      //   }
      // }
      // C NxN
      // NxN C
      af::array tmp1 = af::transpose(posEmbAll);
      af::array tmp = af::moddims(tmp1, af::dim4(imageDim, imageDim, C, 1));
      // N N C
      af::array posEmbAllRes =
          af::resize(tmp, newImgSize, newImgSize, AF_INTERP_BICUBIC);
      for (int i = 0; i < 10; i++) {
        af::array posEmbAllResTMP =
          af::resize(tmp, newImgSize, newImgSize, AF_INTERP_BICUBIC);
          std::cout << "CHECK" << af::allTrue<bool>(posEmbAllResTMP == posEmbAllRes) << std::endl;
      }
      // N_new N_new C
      posEmbAllRes =
          af::moddims(posEmbAllRes, af::dim4(newImgSize * newImgSize, C));
      // N_newxN_new C
      posEmbAllRes = af::transpose(posEmbAllRes);
      // C N_newxN_new
      posembResize_ = fl::Variable(af::join(1, clsEmb, posEmbAllRes), false);
    } else {
      posembResize_ = params_[1];
    }
    af::print("emb", posembResize_.array());
  }
}

std::vector<fl::Variable> ViT::forward(
    const std::vector<fl::Variable>& inputs,
    bool useFp16,
    bool eval,
    bool aug,
    float region,
    bool doShrink,
    float embDropout) {
  // Patching
  auto output = patchEmbedding_->forward(inputs[0]); // H x W x C x B
  int H = output.dims(0);
  int W = output.dims(1);
  if (useFp16) {
    // Avoid running conv2D in fp16 in this case.
    // All the other part of the network can be run in fp16 if compatible.
    output = output.as(f16);
  }
  output = moddims(output, af::dim4(-1, 1, 0, 0)); // T x 1 x C x B
  output = reorder(output, 2, 0, 3, 1); // C x T x B
  auto B = output.dims(2);

  // Prepending the class token
  auto clsToken =
      tile(params_[0], af::dim4(1, 1, B)).as(output.type()); // C x 1 x B
  output = concatenate({clsToken, output}, 1);

  // Positional embedding
  if (usePosEmb_) {
    // if (eval && params_[1].dims(1) != output.dims(1) &&
    //     posembResize_.isempty()) {
    //   af::array posEmbArr = params_[1].array() + 0.;
    //   af::print("posEmbArr", posEmbArr);
    //   // af::eval(posEmbArr);
    //   af::array clsEmb = posEmbArr.col(0) + 0.;
    //   std::cerr << "posEmbArr " << posEmbArr.dims() << std::endl;
    //   af::array posEmbAll = posEmbArr.cols(1, posEmbArr.dims(1) - 1) + 0.;
    //   af::print("posEmbAll", posEmbAll);

    //   int imageDim = std::sqrt(posEmbArr.dims(1) - 1);
    //   int C = posEmbArr.dims(0);
    //   int imageDimNew = std::sqrt(output.dims(1) - 1);
    //   if (imageDim * imageDim != posEmbAll.dims(1)) {
    //     std::cerr << "ERROR in up/down-sampling";
    //   }
    //   if (imageDimNew * imageDimNew != output.dims(1) - 1) {
    //     std::cerr << "ERROR2 in up/down-sampling";
    //   }
    //   // C NxN
    //   // NxN C
    //   af::array tmp1 = af::transpose(posEmbAll);
    //   af::print("tmp1", tmp1);
    //   af::array tmp = af::moddims(tmp1, af::dim4(imageDim, imageDim, C, 1));
    //   af::eval(tmp);
    //   // N N C
    //   af::print("To resize", tmp);
    //   af::array posEmbAllRes =
    //       af::resize(tmp, imageDimNew, imageDimNew, AF_INTERP_BICUBIC);
    //   af::print("Resized version", posEmbAllRes);
    //   // N_new N_new C
    //   posEmbAllRes =
    //       af::moddims(posEmbAllRes, af::dim4(imageDimNew * imageDimNew, C));
    //   // N_newxN_new C
    //   posEmbAllRes = af::transpose(posEmbAllRes);
    //   // C N_newxN_new
    //   posembResize_ = fl::Variable(af::join(1, clsEmb, posEmbAllRes), false);
    // } else if (posembResize_.isempty()) {
    //   posembResize_ = params_[1];
    // }
    fl::Variable posEmb;
    if (!posembResize_.isempty()) {
      posEmb = tile(posembResize_, af::dim4(1, 1, B)).as(output.type());
    } else {
      posEmb = tile(params_[1], af::dim4(1, 1, B)).as(output.type());
    }
    if (train_) {
      posEmb = dropout(posEmb, embDropout);
    }
    // std::cerr << posEmb.dims() << " | " << output.dims() << std::endl;
    output = output + posEmb;
  } else if (useAugPosEmb_) {
    output = augPosEmb_->forward({output}, aug, region, doShrink, H, W).front();
  }
  if (train_) {
    output = dropout(output, pDropout_);
  }
  // Transformers
  for (int i = 0; i < nLayers_; ++i) {
    output = transformers_[i]->forward({output}).front();
  }
  // Linear
  output = ln_->forward(output); // C x T x B
  output = reorder(output, 0, 2, 1, 3).slice(0); // C x B x 1
  output = linearOut_->forward(output);
  return {output};
}

std::string ViT::prettyString() const {
  std::ostringstream ss;
  ss << "ViT (" << nClasses_ << " classes) with " << nLayers_
     << " Transformers:\n";
  for (const auto& transformers : transformers_) {
    ss << transformers->prettyString() << "\n";
  }
  return ss.str();
}

} // namespace image
} // namespace ext
} // namespace fl
