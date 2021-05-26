/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/runtime/Attention.h"

namespace fl {
namespace pkg {
namespace speech {

std::shared_ptr<AttentionBase> createAttention() {
  std::shared_ptr<AttentionBase> attention;
  if (FLAGS_attention == fl::pkg::speech::kContentAttention) {
    attention = std::make_shared<ContentAttention>();
  } else if (FLAGS_attention == fl::pkg::speech::kKeyValueAttention) {
    attention = std::make_shared<ContentAttention>(true);
  } else if (FLAGS_attention == fl::pkg::speech::kNeuralContentAttention) {
    attention = std::make_shared<NeuralContentAttention>(FLAGS_encoderdim);
  } else if (FLAGS_attention == fl::pkg::speech::kSimpleLocationAttention) {
    attention = std::make_shared<SimpleLocationAttention>(FLAGS_attnconvkernel);
  } else if (FLAGS_attention == fl::pkg::speech::kLocationAttention) {
    attention = std::make_shared<LocationAttention>(
        FLAGS_encoderdim, FLAGS_attnconvkernel);
  } else if (FLAGS_attention == fl::pkg::speech::kNeuralLocationAttention) {
    attention = std::make_shared<NeuralLocationAttention>(
        FLAGS_encoderdim,
        FLAGS_attndim,
        FLAGS_attnconvchannel,
        FLAGS_attnconvkernel);
  } // is it fine for transformer criterion?
  else if (FLAGS_attention == fl::pkg::speech::kMultiHeadContentAttention) {
    attention = std::make_shared<MultiHeadContentAttention>(
        FLAGS_encoderdim, FLAGS_numattnhead);
  } else if (
      FLAGS_attention == fl::pkg::speech::kMultiHeadKeyValueContentAttention) {
    attention = std::make_shared<MultiHeadContentAttention>(
        FLAGS_encoderdim, FLAGS_numattnhead, true);
  } else if (FLAGS_attention == fl::pkg::speech::kMultiHeadSplitContentAttention) {
    attention = std::make_shared<MultiHeadContentAttention>(
        FLAGS_encoderdim, FLAGS_numattnhead, false, true);
  } else if (
      FLAGS_attention ==
      fl::pkg::speech::kMultiHeadKeyValueSplitContentAttention) {
    attention = std::make_shared<MultiHeadContentAttention>(
        FLAGS_encoderdim, FLAGS_numattnhead, true, true);
  } else {
    throw std::runtime_error("Unimplmented attention: " + FLAGS_attention);
  }
  return attention;
}

std::shared_ptr<WindowBase> createAttentionWindow() {
  std::shared_ptr<WindowBase> window;
  if (FLAGS_attnWindow == fl::pkg::speech::kNoWindow) {
    window = nullptr;
  } else if (FLAGS_attnWindow == fl::pkg::speech::kMedianWindow) {
    window = std::make_shared<MedianWindow>(
        FLAGS_leftWindowSize, FLAGS_rightWindowSize);
  } else if (FLAGS_attnWindow == fl::pkg::speech::kStepWindow) {
    window = std::make_shared<StepWindow>(
        FLAGS_minsil, FLAGS_maxsil, FLAGS_minrate, FLAGS_maxrate);
  } else if (FLAGS_attnWindow == fl::pkg::speech::kSoftWindow) {
    window = std::make_shared<SoftWindow>(
        FLAGS_softwstd, FLAGS_softwrate, FLAGS_softwoffset);
  } else if (FLAGS_attnWindow == fl::pkg::speech::kSoftPretrainWindow) {
    window = std::make_shared<SoftPretrainWindow>(FLAGS_softwstd);
  } else {
    throw std::runtime_error("Unimplmented window: " + FLAGS_attnWindow);
  }
  return window;
}
} // namespace speech
} // namespace pkg
} // namespace fl
