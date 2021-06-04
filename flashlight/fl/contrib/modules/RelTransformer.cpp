/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/contrib/modules/RelTransformer.h"
#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

namespace {
fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std, af::dtype::f32, true);
}

fl::Variable relMultiheadAttention(
    const fl::Variable& query,
    const fl::Variable& key,
    const fl::Variable& value,
    const fl::Variable& rKey,
    const fl::Variable& rWbias,
    const fl::Variable& rRbias,
    const fl::Variable& padMask,
    const int32_t nHeads,
    const double pDropout) {
  int32_t bsz = query.dims(2);
  int32_t modelDim = query.dims(1);
  int32_t headDim = modelDim / nHeads;

  auto q = moddims(query, af::dim4(-1, headDim, nHeads * bsz));
  auto k = moddims(key, af::dim4(-1, headDim, nHeads * bsz));
  auto v = moddims(value, af::dim4(-1, headDim, nHeads * bsz));

  // auto rQ = moddims(rQuery, af::dim4(-1, headDim, nHeads * bsz));
  auto rK = moddims(rKey, af::dim4(-1, headDim, nHeads * bsz));

  // take for rQ last position which should be WRQ *[sin(0) cos(0)]
  // auto rwQ = q + fl::tileAs(rQ.row(rQ.dims(0) - 1), q);
  auto rwQ =
      q +
      fl::moddims(
          fl::tileAs(
              fl::moddims(rWbias.as(q.type()), af::dim4(1, headDim, nHeads)),
              af::dim4(q.dims(0), headDim, nHeads, bsz)),
          af::dim4(-1, headDim, nHeads * bsz));
  auto AC = matmulNT(rwQ, k);
  auto rrQ =
      q +
      fl::moddims(
          fl::tileAs(
              fl::moddims(rRbias.as(q.type()), af::dim4(1, headDim, nHeads)),
              af::dim4(q.dims(0), headDim, nHeads, bsz)),
          af::dim4(-1, headDim, nHeads * bsz));
  fl::Variable BD = matmulNT(rrQ, rK);
  fl::Variable pad =
      fl::Variable(
          af::constant(0, af::dim4(BD.dims(0), 1, BD.dims(2), BD.dims(3))),
          false)
          .as(BD.type());
  auto relBD = fl::concatenate({BD, pad}, 1);
  relBD = fl::moddims(
      relBD, af::dim4(BD.dims(1) + 1, BD.dims(0), BD.dims(2), BD.dims(3)));
  relBD = fl::moddims(relBD.rows(1, BD.dims(1)), BD.dims());
  // rel shift
  // auto relBD =
  //     fl::moddims(BD, af::dim4(BD.dims(1), BD.dims(0), BD.dims(2),
  //     BD.dims(3)));
  // relBD = fl::moddims(
  //     relBD.rows(1, relBD.dims(0) - 1),
  //     af::dim4(BD.dims(0), BD.dims(1) - 1, BD.dims(2), BD.dims(3)));
  auto scores = AC + relBD;
  scores = scores / std::sqrt(float(headDim));
  fl::Variable totalMask;
  if (!padMask.isempty()) {
    if (padMask.dims(0) != query.dims(0)) {
      throw std::invalid_argument(
          "multiheadAttention: invalid padding mask size");
    }
    auto padMaskTile = moddims(padMask, af::dim4(1, padMask.dims(0), 1, bsz));
    padMaskTile = tileAs(
        padMaskTile, af::dim4(padMask.dims(0), padMask.dims(0), nHeads, bsz));
    scores = scores +
        moddims(padMaskTile.as(scores.type()),
                af::dim4(padMask.dims(0), padMask.dims(0), nHeads * bsz));
  }
  scores = fl::softmax(scores, 1);
  auto attn = fl::dropout(scores, pDropout);
  auto result = matmul(attn.as(v.type()), v);
  result = moddims(result, af::dim4(-1, headDim * nHeads, bsz));
  return result;
}
} // namespace

namespace fl {

RelTransformer::RelTransformer(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    float pDropout,
    float pLayerdrop,
    bool preLN,
    bool augment)
    : nHeads_(nHeads),
      pDropout_(pDropout),
      pLayerdrop_(pLayerdrop),
      preLN_(preLN),
      augment_(augment),
      w1_(std::make_shared<Linear>(transformerInitLinear(modelDim, mlpDim))),
      w2_(std::make_shared<Linear>(transformerInitLinear(mlpDim, modelDim))),
      wq_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wk_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      rk_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      rWbias_(transformerInitLinear(nHeads, headDim)),
      rRbias_(transformerInitLinear(nHeads, headDim)),
      wv_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads))),
      wf_(std::make_shared<Linear>(
          transformerInitLinear(headDim * nHeads, modelDim))),
      norm1_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))),
      norm2_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))) {
  add(w1_);
  add(w2_);
  add(wq_);
  add(wk_);
  add(rk_);
  params_.emplace_back(rWbias_);
  params_.emplace_back(rRbias_);
  add(wv_);
  add(wf_);
  add(norm1_);
  add(norm2_);
}

Variable RelTransformer::mlp(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  return (*w2_)(dropout(relu((*w1_)(input)), pDropout));
}

Variable RelTransformer::selfAttention(const std::vector<Variable>& input) {
  // input [C, T, B] relpos [C, T + 1, B] padMask
  auto encoderInput = input[0];
  auto posInput = input[1];
  int t = encoderInput.dims(1), bsz = encoderInput.dims(2);
  double pDrop = train_ ? pDropout_ : 0.0;

  // [T, C, B]
  auto q = transpose((*wq_)(encoderInput));
  auto k = transpose((*wk_)(encoderInput));
  auto v = transpose((*wv_)(encoderInput));

  // [T + 1, C, B]
  // auto rQ = transpose((*wq_)(posInput));
  // [T, C, B]
  auto rK = transpose((*rk_)(posInput));

  // time x batch
  fl::Variable padMask;
  if (!input.back().isempty()) {
    auto padMaskArr = input.back().array();
    padMaskArr =
        af::resize(padMaskArr, encoderInput.dims(1), encoderInput.dims(2));
    padMask = fl::Variable(af::log(padMaskArr), false);
  }

  auto result = relMultiheadAttention(
      q, k, v, rK, rWbias_, rRbias_, padMask, nHeads_, pDrop);
  result = (*wf_)(transpose(result));

  return result;
}

std::vector<Variable> RelTransformer::forward(
    const std::vector<Variable>& input) {
  // previous step[optionally], input, padMask
  // padMask should be empty if previous step is provided
  // padMask is expected to have "1" on the used positions and "0" on padded
  // positions
  if (input.size() < 3) {
    throw std::invalid_argument(
        "Invalid inputs for transformer block: there should be at least input and mask");
  }
  auto x = input[0];
  if (!input.back().isempty() && x.dims(2) != input.back().dims(1)) {
    throw std::invalid_argument(
        "Invalid inputs for transformer block: input and Mask batch sizes are different");
  }
  float f = 1.0;
  if (train_ && (af::randu(1).scalar<float>() < pLayerdrop_)) {
    f = 0.0;
  }
  if (preLN_) {
    auto h = (f * (*norm1_)(selfAttention(input))).as(x.type()) + x;
    return {f * (*norm2_)(mlp(h)).as(h.type()) + h};
  } else {
    auto h = (*norm1_)((f * selfAttention(input)).as(x.type()) + x);
    return {(*norm2_)((f * mlp(h)).as(h.type()) + h)};
  }
}

std::string RelTransformer::prettyString() const {
  std::ostringstream ss;
  ss << "RelTransformer (nHeads: " << nHeads_ << "), "
     << "(pDropout: " << pDropout_ << "), "
     << "(pLayerdrop: " << pLayerdrop_ << "), "
     << "(preLayerNorm: " << preLN_ << ")";
  return ss.str();
}

RelTransformer::RelTransformer() {}

} // namespace fl
