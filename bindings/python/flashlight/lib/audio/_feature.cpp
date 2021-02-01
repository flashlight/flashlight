/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "flashlight/lib/audio/feature/Ceplifter.h"
#include "flashlight/lib/audio/feature/Dct.h"
#include "flashlight/lib/audio/feature/Derivatives.h"
#include "flashlight/lib/audio/feature/Dither.h"
#include "flashlight/lib/audio/feature/FeatureParams.h"
#include "flashlight/lib/audio/feature/Mfcc.h"
#include "flashlight/lib/audio/feature/Mfsc.h"
#include "flashlight/lib/audio/feature/PowerSpectrum.h"
#include "flashlight/lib/audio/feature/PreEmphasis.h"
#include "flashlight/lib/audio/feature/SpeechUtils.h"
#include "flashlight/lib/audio/feature/TriFilterbank.h"
#include "flashlight/lib/audio/feature/Windowing.h"

namespace py = pybind11;
using namespace pybind11::literals;

using WindowType = fl::lib::audio::WindowType;
using FrequencyScale = fl::lib::audio::FrequencyScale;
using FeatureParams = fl::lib::audio::FeatureParams;

using Ceplifter = fl::lib::audio::Ceplifter;
using Dct = fl::lib::audio::Dct;
using Derivatives = fl::lib::audio::Derivatives;
using Dither = fl::lib::audio::Dither;
using Mfcc = fl::lib::audio::Mfcc;
using Mfsc = fl::lib::audio::Mfsc;
using PowerSpectrum = fl::lib::audio::PowerSpectrum;
using PreEmphasis = fl::lib::audio::PreEmphasis;
using TriFilterbank = fl::lib::audio::TriFilterbank;
using Windowing = fl::lib::audio::Windowing;

PYBIND11_MODULE(flashlight_lib_audio_feature, m) {
  py::enum_<WindowType>(m, "WindowType")
      .value("HAMMING", WindowType::HAMMING)
      .value("HANNING", WindowType::HANNING);
  py::enum_<FrequencyScale>(m, "FrequencyScale")
      .value("MEL", FrequencyScale::MEL)
      .value("LINEAR", FrequencyScale::LINEAR)
      .value("LOG10", FrequencyScale::LOG10);
  py::class_<FeatureParams>(m, "FeatureParams")
      .def(
          py::init<
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              WindowType,
              float,
              float,
              float,
              bool,
              bool,
              bool,
              bool>(),
          "sampling_freq"_a = 16000,
          "frame_size_ms"_a = 25,
          "frame_stride_ms"_a = 10,
          "num_filterbank_chans"_a = 23,
          "low_freq_filterbank"_a = 0,
          "high_freq_filterbank"_a = -1,
          "num_cepstral_coeffs"_a = 13,
          "lifter_param"_a = 22,
          "delta_window"_a = 2,
          "acc_window"_a = 2,
          "window_type"_a = WindowType::HAMMING,
          "preem_coef"_a = 0.97,
          "mel_floor"_a = 1.0,
          "dither_val"_a = 0.0,
          "use_power"_a = true,
          "use_energy"_a = true,
          "raw_energy"_a = true,
          "zero_mean_frame"_a = true)
      .def("num_frame_size_samples", &FeatureParams::numFrameSizeSamples)
      .def("num_frame_stride_samples", &FeatureParams::numFrameStrideSamples)
      .def("n_fft", &FeatureParams::nFft)
      .def("filter_freq_response_len", &FeatureParams::filterFreqResponseLen)
      .def("pow_spec_feat_sz", &FeatureParams::powSpecFeatSz)
      .def("mfsc_feat_sz", &FeatureParams::mfscFeatSz)
      .def("mfcc_feat_sz", &FeatureParams::mfccFeatSz)
      .def("num_frames", &FeatureParams::numFrames)
      .def_readwrite("sampling_freq", &FeatureParams::samplingFreq)
      .def_readwrite("frame_size_ms", &FeatureParams::frameSizeMs)
      .def_readwrite("frame_stride_ms", &FeatureParams::frameStrideMs)
      .def_readwrite("num_filterbank_chans", &FeatureParams::numFilterbankChans)
      .def_readwrite("low_freq_filterbank", &FeatureParams::lowFreqFilterbank)
      .def_readwrite("high_freq_filterbank", &FeatureParams::highFreqFilterbank)
      .def_readwrite("num_cepstral_coeffs", &FeatureParams::numCepstralCoeffs)
      .def_readwrite("lifter_params", &FeatureParams::lifterParam)
      .def_readwrite("delta_window", &FeatureParams::deltaWindow)
      .def_readwrite("acc_window", &FeatureParams::accWindow)
      .def_readwrite("window_type", &FeatureParams::windowType)
      .def_readwrite("preem_coef", &FeatureParams::preemCoef)
      .def_readwrite("mel_floor", &FeatureParams::melFloor)
      .def_readwrite("dither_val", &FeatureParams::ditherVal)
      .def_readwrite("use_power", &FeatureParams::usePower)
      .def_readwrite("use_energy", &FeatureParams::useEnergy)
      .def_readwrite("raw_energy", &FeatureParams::rawEnergy)
      .def_readwrite("zero_mean_frame", &FeatureParams::zeroMeanFrame);

  py::class_<Ceplifter>(m, "Ceplifter")
      .def(py::init<int64_t, int64_t>(), "num_filters"_a, "lifter_param"_a)
      .def("apply", &Ceplifter::apply, "input"_a)
      .def("apply_in_place", &Ceplifter::applyInPlace, "input"_a);
  py::class_<Dct>(m, "Dct")
      .def(py::init<int64_t, int64_t>(), "num_filters"_a, "num_ceps"_a)
      .def("apply", &Dct::apply, "input"_a);
  py::class_<Derivatives>(m, "Derivatives")
      .def(py::init<int64_t, int64_t>(), "delta_window"_a, "acc_window"_a)
      .def("apply", &Derivatives::apply, "input"_a, "num_feat"_a);
  py::class_<Dither>(m, "Dither")
      .def(py::init<float>(), "dither_val"_a)
      .def("apply", &Dither::apply, "input"_a)
      .def("apply_in_place", &Dither::applyInPlace, "input"_a);
  py::class_<Mfcc>(m, "Mfcc")
      .def(py::init<const FeatureParams&>(), "params"_a)
      .def("apply", &Mfcc::apply, "input"_a)
      .def("batch_apply", &Mfcc::batchApply, "input"_a, "batch_sz"_a)
      .def("output_size", &Mfcc::outputSize, "input_sz"_a)
      .def("get_feature_params", &Mfcc::getFeatureParams);
  py::class_<Mfsc>(m, "Mfsc")
      .def(py::init<const FeatureParams&>(), "params"_a)
      .def("apply", &Mfsc::apply, "input"_a)
      .def("batch_apply", &Mfsc::batchApply, "input"_a, "batch_sz"_a)
      .def("output_size", &Mfsc::outputSize, "input_sz"_a)
      .def("get_feature_params", &Mfsc::getFeatureParams);
  py::class_<PowerSpectrum>(m, "PowerSpectrum")
      .def(py::init<const FeatureParams&>(), "params"_a)
      .def("apply", &PowerSpectrum::apply, "input"_a)
      .def("batch_apply", &PowerSpectrum::batchApply, "input"_a, "batch_sz"_a)
      .def("output_size", &PowerSpectrum::outputSize, "input_sz"_a)
      .def("get_feature_params", &PowerSpectrum::getFeatureParams);
  py::class_<PreEmphasis>(m, "PreEmphasis")
      .def(py::init<float, int64_t>(), "alpha"_a, "N"_a)
      .def("apply", &PreEmphasis::apply, "input"_a)
      .def("apply_in_place", &PreEmphasis::applyInPlace, "input"_a);
  py::class_<TriFilterbank>(m, "TriFilterbank")
      .def(
          py::init<
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              int64_t,
              FrequencyScale>(),
          "num_filters"_a,
          "filter_len"_a,
          "sampling_freq"_a,
          "low_freq"_a = 0,
          "high_freq"_a = -1,
          "freq_scale"_a = FrequencyScale::MEL)
      .def("apply", &TriFilterbank::apply, "input"_a, "mel_floor"_a = 0.0)
      .def("filterbank", &TriFilterbank::filterbank);
  py::class_<Windowing>(m, "Windowing")
      .def(py::init<int64_t, WindowType>(), "N"_a, "window"_a)
      .def("apply", &Windowing::apply, "input"_a)
      .def("apply_in_place", &Windowing::applyInPlace, "input"_a);

  m.def("frame_signal", fl::lib::audio::frameSignal, "input"_a, "params"_a);
  m.def("cblas_gemm", fl::lib::audio::cblasGemm, "A"_a, "B"_a, "n"_a, "k"_a);
}
