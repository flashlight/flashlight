/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/VisionTransformer.h"

#include <cmath>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace {

const float pi = std::acos(-1);
const float geluConst1 = std::sqrt(2 / pi);
const float geluConst2 = 0.044715;

} // namespace

namespace fl {

VisionTransformer::VisionTransformer(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    float pDropout,
    float pLayerdrop)
    : nHeads_(nHeads),
      pDropout_(pDropout),
      pLayerdrop_(pLayerdrop),
      w1_(std::make_shared<Linear>(initLinear(modelDim, mlpDim))),
      w2_(std::make_shared<Linear>(initLinear(mlpDim, modelDim))),
      wq_(std::make_shared<Linear>(
          initLinear(modelDim, headDim * nHeads),
          fl::constant(0., headDim * nHeads, 1))),
      wk_(std::make_shared<Linear>(
          initLinear(modelDim, headDim * nHeads),
          fl::constant(0., headDim * nHeads, 1))),
      wv_(std::make_shared<Linear>(
          initLinear(modelDim, headDim * nHeads),
          fl::constant(0., headDim * nHeads, 1))),
      // wq_(std::make_shared<Linear>(initLinear(modelDim, headDim * nHeads))),
      // wk_(std::make_shared<Linear>(initLinear(modelDim, headDim * nHeads))),
      // wv_(std::make_shared<Linear>(initLinear(modelDim, headDim * nHeads))),
      wf_(std::make_shared<Linear>(initLinear(headDim * nHeads, modelDim))),
      norm1_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}), 1e-6)),
      norm2_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}), 1e-6)) {
  add(w1_);
  add(w2_);
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
  add(norm1_);
  add(norm2_);
}

Variable VisionTransformer::gelu(const Variable& input) {
  // https://arxiv.org/pdf/1606.08415.pdf
  return 0.5 * input *
      (1 + tanh(geluConst1 * (input + geluConst2 * pow(input, 3))));
}

Variable VisionTransformer::mlp(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  auto output = (*w1_)(input);
  output = gelu(output);
  output = dropout(output, pDropout);
  output = (*w2_)(output);
  output = dropout(output, pDropout);

  return output;
}

Variable VisionTransformer::selfAttention(const Variable& x) {
  int n = x.dims(1), bsz = x.dims(2);
  double pDrop = train_ ? pDropout_ : 0.0;

  auto q = transpose((*wq_)(x));
  auto k = transpose((*wk_)(x));
  auto v = transpose((*wv_)(x));

  Variable mask, posEmb, padMask;
  auto result =
      multiheadAttention(q, k, v, posEmb, mask, padMask, nHeads_, pDrop);
  result = (*wf_)(transpose(result));
  result = dropout(result, pDrop);

  return result;
}

std::vector<Variable> VisionTransformer::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 1) {
    throw std::runtime_error("VisionTransformer forward, >1 inputs");
  }
  auto x = inputs.front();

  float f = 1.0;
  if (train_ && (af::randu(1).scalar<float>() < pLayerdrop_)) {
    f = 0.0;
  }

  auto output = x + f * selfAttention((*norm1_)(x));
  output = output + f * mlp((*norm2_)(output));
  return {output};
}

std::string VisionTransformer::prettyString() const {
  std::ostringstream ss;
  ss << "VisionTransformer (nHeads: " << nHeads_ << "), "
     << "(pDropout: " << pDropout_ << "), "
     << "(pLayerdrop: " << pLayerdrop_ << "), ";
  return ss.str();
}

VisionTransformer::VisionTransformer() {}

fl::Variable VisionTransformer::initLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std, af::dtype::f32, true);

  // TODO: rigours truncated normal
  // auto arr = af::normal(af::dim4(outDim, inDim), 0.02);
  // arr = af::clamp(arr, -2., 2.);
  // auto perterb = af::randu(arr.dims()) * 1e-8 - 5e-9;
  // return fl::Variable(arr + perterb, false);
}

} // namespace fl
