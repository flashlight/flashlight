/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/runtime/Attention.h"

namespace fl {
namespace app {
namespace asr {

std::shared_ptr<AttentionBase> createAttention() {
  std::shared_ptr<AttentionBase> attention;
  if (FLAGS_attention == fl::app::asr::kContentAttention) {
    attention = std::make_shared<ContentAttention>();
  } else if (FLAGS_attention == fl::app::asr::kKeyValueAttention) {
    attention = std::make_shared<ContentAttention>(true);
  } else if (FLAGS_attention == fl::app::asr::kNeuralContentAttention) {
    attention = std::make_shared<NeuralContentAttention>(FLAGS_encoderdim);
  } else if (FLAGS_attention == fl::app::asr::kSimpleLocationAttention) {
    attention = std::make_shared<SimpleLocationAttention>(FLAGS_attnconvkernel);
  } else if (FLAGS_attention == fl::app::asr::kLocationAttention) {
    attention = std::make_shared<LocationAttention>(
        FLAGS_encoderdim, FLAGS_attnconvkernel);
  } else if (FLAGS_attention == fl::app::asr::kNeuralLocationAttention) {
    attention = std::make_shared<NeuralLocationAttention>(
        FLAGS_encoderdim,
        FLAGS_attndim,
        FLAGS_attnconvchannel,
        FLAGS_attnconvkernel);
  } // is it fine for transformer criterion?
  else if (FLAGS_attention == fl::app::asr::kMultiHeadContentAttention) {
    attention = std::make_shared<MultiHeadContentAttention>(
        FLAGS_encoderdim, FLAGS_numattnhead);
  } else if (
      FLAGS_attention == fl::app::asr::kMultiHeadKeyValueContentAttention) {
    attention = std::make_shared<MultiHeadContentAttention>(
        FLAGS_encoderdim, FLAGS_numattnhead, true);
  } else if (FLAGS_attention == fl::app::asr::kMultiHeadSplitContentAttention) {
    attention = std::make_shared<MultiHeadContentAttention>(
        FLAGS_encoderdim, FLAGS_numattnhead, false, true);
  } else if (
      FLAGS_attention ==
      fl::app::asr::kMultiHeadKeyValueSplitContentAttention) {
    attention = std::make_shared<MultiHeadContentAttention>(
        FLAGS_encoderdim, FLAGS_numattnhead, true, true);
  } else {
    throw std::runtime_error("Unimplmented attention: " + FLAGS_attention);
  }
  return attention;
}

std::shared_ptr<WindowBase> createAttentionWindow() {
  std::shared_ptr<WindowBase> window;
  if (FLAGS_attnWindow == fl::app::asr::kNoWindow) {
    window = nullptr;
  } else if (FLAGS_attnWindow == fl::app::asr::kMedianWindow) {
    window = std::make_shared<MedianWindow>(
        FLAGS_leftWindowSize, FLAGS_rightWindowSize);
  } else if (FLAGS_attnWindow == fl::app::asr::kStepWindow) {
    window = std::make_shared<StepWindow>(
        FLAGS_minsil, FLAGS_maxsil, FLAGS_minrate, FLAGS_maxrate);
  } else if (FLAGS_attnWindow == fl::app::asr::kSoftWindow) {
    window = std::make_shared<SoftWindow>(
        FLAGS_softwstd, FLAGS_softwrate, FLAGS_softwoffset);
  } else if (FLAGS_attnWindow == fl::app::asr::kSoftPretrainWindow) {
    window = std::make_shared<SoftPretrainWindow>(FLAGS_softwstd);
  } else {
    throw std::runtime_error("Unimplmented window: " + FLAGS_attnWindow);
  }
  return window;
}
} // namespace asr
} // namespace app
} // namespace fl
