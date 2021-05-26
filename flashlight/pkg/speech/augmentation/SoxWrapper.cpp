/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/augmentation/SoxWrapper.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

#include <glog/logging.h>
#include <sox.h>

#include "flashlight/pkg/speech/data/Sound.h"
#include "flashlight/fl/common/Logging.h"

namespace fl {
namespace pkg {
namespace speech {
namespace sfx {

namespace {

struct SoxData {
  std::vector<float>* data;
  size_t index = 0;
};

static int outputFlow(
    sox_effect_t* effp LSX_UNUSED,
    sox_sample_t const* ibuf,
    sox_sample_t* obuf LSX_UNUSED,
    size_t* isamp,
    size_t* osamp) {
  if (*isamp) {
    auto priv = static_cast<SoxData*>(effp->priv);

    int i = 0;
    for (; i < *isamp; ++i) {
      SOX_SAMPLE_LOCALS;
      priv->data->push_back(SOX_SAMPLE_TO_FLOAT_32BIT(ibuf[i], effp->clips));
    }

    if (i != *isamp) {
      LOG(ERROR) << "outputFlow number of bytes written=" << i
                 << " expected=" << *isamp
                 << " priv->data->size()=" << priv->data->size();
      return SOX_EOF;
    }
  }

  *osamp = 0;
  return SOX_SUCCESS;
}

int inputDrain(sox_effect_t* effp, sox_sample_t* obuf, size_t* osamp) {
  auto priv = static_cast<SoxData*>(effp->priv);

  int i = 0;
  for (; i < *osamp && priv->index < priv->data->size(); ++i, ++priv->index) {
    SOX_SAMPLE_LOCALS;
    obuf[i] =
        SOX_FLOAT_32BIT_TO_SAMPLE(priv->data->at(priv->index), effp->clips);
  }
  *osamp = i;
  return *osamp ? SOX_SUCCESS : SOX_EOF;
}

std::unique_ptr<sox_signalinfo_t> createSignalInfo(size_t sampleRate) {
  auto sigInfo = std::make_unique<sox_signalinfo_t>();
  *sigInfo = {
      .rate = (sox_rate_t)sampleRate,
      .channels = 1, // Sounds effects are limited to single channel
      .precision = 16, // Any valid value is ok here.
      .length = 0,
      .mult = nullptr};
  return sigInfo;
}

} // namespace

std::unique_ptr<SoxWrapper> SoxWrapper::instance_;

SoxWrapper::SoxWrapper(size_t sampleRate)
    : signalInfo_(createSignalInfo(sampleRate)) {
  FL_SOX_CHECK(sox_init());
}

SoxWrapper::~SoxWrapper() {
  sox_quit();
}

SoxWrapper* SoxWrapper::instance(size_t sampleRate /* =16000*/) {
  if (!instance_) {
    auto s = new SoxWrapper(sampleRate);
    instance_.reset(s);
  }
  return instance_.get();
}

void SoxWrapper::applyAndFreeEffect(
    std::vector<float>& signal,
    sox_effect_t* effect) const {
  sox_effects_chain_t* chain = createChain();
  addInput(chain, &signal);
  FL_SOX_CHECK(
      sox_add_effect(chain, effect, signalInfo_.get(), signalInfo_.get()));
  free(effect);
  std::vector<float> augmented;
  addOutput(chain, &augmented);

  sox_flow_effects(chain, nullptr, nullptr);

  sox_delete_effects_chain(chain);
  signal.swap(augmented);
}

void SoxWrapper::addInput(
    sox_effects_chain_t* chain,
    std::vector<float>* signal) const {
  const static sox_effect_handler_t handler{
      /*name=*/"input",
      /*usage=*/nullptr,
      /*flags=*/SOX_EFF_MCHAN,
      /*getopts=*/nullptr,
      /*start=*/nullptr,
      /*flow=*/nullptr,
      /*drain=*/inputDrain,
      /*stop=*/nullptr,
      /*kill=*/nullptr,
      /*priv_size=*/sizeof(SoxData)};
  sox_effect_t* e = nullptr;
  FL_SOX_CHECK(e = sox_create_effect(&handler));
  auto input = (SoxData*)e->priv;
  input->data = signal;
  input->index = 0;
  FL_SOX_CHECK(sox_add_effect(chain, e, signalInfo_.get(), signalInfo_.get()));
  free(e);
}

void SoxWrapper::addOutput(
    sox_effects_chain_t* chain,
    std::vector<float>* emptyBuf) const {
  const static sox_effect_handler_t handler = {
      /*name=*/"output",
      /*usage=*/nullptr,
      /*flags=*/SOX_EFF_MCHAN,
      /*getopts=*/nullptr,
      /*start=*/nullptr,
      /*flow=*/outputFlow,
      /*drain=*/nullptr,
      /*stop=*/nullptr,
      /*kill=*/nullptr,
      /*priv_size=*/sizeof(SoxData)};
  sox_effect_t* e = nullptr;
  FL_SOX_CHECK(e = sox_create_effect(&handler));
  auto output = (SoxData*)e->priv;
  output->data = emptyBuf;
  FL_SOX_CHECK(sox_add_effect(chain, e, signalInfo_.get(), signalInfo_.get()));
  free(e);
}

void SoxWrapper::addAndFreeEffect(
    sox_effects_chain_t* chain,
    sox_effect_t* effect) const {
  FL_SOX_CHECK(
      sox_add_effect(chain, effect, signalInfo_.get(), signalInfo_.get()));
  free(effect);
}

sox_effects_chain_t* SoxWrapper::createChain() const {
  const static sox_encodinginfo_t encoding = {
      .encoding = SOX_ENCODING_FLOAT,
      .bits_per_sample = 0,
      .compression = HUGE_VAL, // no compression
      .reverse_bytes = sox_option_no,
      .reverse_nibbles = sox_option_no,
      .reverse_bits = sox_option_no,
      .opposite_endian = sox_false};
  sox_effects_chain_t* chain = nullptr;
  FL_SOX_CHECK(chain = sox_create_effects_chain(&encoding, &encoding));
  return chain;
}

namespace detail {

void check(bool success, const char* msg, const char* file, int line) {
  if (!success) {
    std::stringstream ss;
    ss << file << ':' << line << "] libsox error when executing: " << msg;
    LOG(ERROR) << ss.str();
    throw std::runtime_error(ss.str());
  }
}

void check(int status, const char* msg, const char* file, int line) {
  if (status != SOX_SUCCESS) {
    std::stringstream ss;
    ss << file << ':' << line << "] libsox error: " << status
       << " when executing: " << msg;
    LOG(ERROR) << ss.str();
    throw std::runtime_error(ss.str());
  }
}

void check(const void* ptr, const char* msg, const char* file, int line) {
  if (!ptr) {
    std::stringstream ss;
    ss << file << ':' << line
       << "] libsox failed to allocate when executing: " << msg;
    LOG(ERROR) << ss.str();
    throw std::runtime_error(ss.str());
  }
}

} // namespace detail

} // namespace sfx
} // namespace speech
} // namespace pkg
} // namespace fl
