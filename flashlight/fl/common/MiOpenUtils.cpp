/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/common/MiOpenUtils.h"

namespace fl {
namespace miopen {

std::string PrettyString(miopenStatus_t status) {
  switch (status) {
    case miopenStatusSuccess:
      return "miopenStatusSuccess";
    case miopenStatusNotInitialized:
      return "miopenStatusNotInitialized";
    case miopenStatusAllocFailed:
      return "miopenStatusAllocFailed";
    case miopenStatusBadParm:
      return "miopenStatusBadParm";
    case miopenStatusInternalError:
      return "miopenStatusInternalError";
    case miopenStatusInvalidValue:
      return "miopenStatusInvalidValue";
    case miopenStatusNotImplemented:
      return "miopenStatusNotImplemented";
    case miopenStatusUnknownError:
      return "miopenStatusUnknownError";
    default:
      std::stringstream ss;
      ss << "Unknown miopen status " << static_cast<int>(status);
      return ss.str();
  }
}

std::string PrettyString(miopenConvFwdAlgorithm_t algorithm) {
  switch (algorithm) {
    case miopenConvolutionFwdAlgoGEMM:
      return "GEMM";
    case miopenConvolutionFwdAlgoDirect:
      return "Direct";
    case miopenConvolutionFwdAlgoFFT:
      return "FFT";
    case miopenConvolutionFwdAlgoWinograd:
      return "Winograd";
    case miopenConvolutionFwdAlgoImplicitGEMM:
      return "Implicit GEMM";
    default:
      std::stringstream ss;
      ss << "Unknown MiOpen forward convolution algorithm "
         << static_cast<int>(algorithm);
      return ss.str();
  }
}

std::string PrettyString(miopenConvBwdWeightsAlgorithm_t algorithm) {
  switch (algorithm) {
    case miopenConvolutionBwdWeightsAlgoGEMM:
      return "GEMM";
    case miopenConvolutionBwdWeightsAlgoDirect:
      return "Direct";
    case miopenConvolutionBwdWeightsAlgoWinograd:
      return "Winograd";
    case miopenConvolutionBwdWeightsAlgoImplicitGEMM:
      return "Implicit GEMM";
    default:
      std::stringstream ss;
      ss << "Unknown MiOpen backward convolution algorithm "
         << static_cast<int>(algorithm);
      return ss.str();
  }
}

std::string PrettyString(miopenConvBwdDataAlgorithm_t algorithm) {
  switch (algorithm) {
    case miopenConvolutionBwdDataAlgoGEMM:
      return "GEMM";
    case miopenConvolutionBwdDataAlgoDirect:
      return "Direct";
    case miopenConvolutionBwdDataAlgoFFT:
      return "FFT";
    case miopenConvolutionBwdDataAlgoWinograd:
      return "Winograd";
    case miopenTransposeBwdDataAlgoGEMM:
      return "Transpose GEMM";
    case miopenConvolutionBwdDataAlgoImplicitGEMM:
      return "Implicit GEMM";
    default:
      std::stringstream ss;
      ss << "Unknown MiOpen backward data convolution algorithm"
         << static_cast<int>(algorithm);
      return ss.str();
  }
}

std::string PrettyString(miopenConvAlgorithm_t algorithm) {
  switch (algorithm) {
    case miopenConvolutionAlgoGEMM:
      return "GEMM";
    case miopenConvolutionAlgoDirect:
      return "Direct";
    case miopenConvolutionAlgoFFT:
      return "FFT";
    case miopenConvolutionAlgoWinograd:
      return "Winograd";
    case miopenConvolutionAlgoImplicitGEMM:
      return "Implicit GEMM";
    default:
      std::stringstream ss;
      ss << "Unknown MiOpen convolution algorithm"
         << static_cast<int>(algorithm);
      return ss.str();
  }
}

std::string PrettyString(miopenConvAlgoPerf_t algorithm) {
  std::stringstream ss;
  ss << "miopenConvAlgoPerf time=" << algorithm.time
     << " memory=" << algorithm.memory;
  return ss.str();
}

std::string PrettyString(miopenConvSolution_t algorithm) {
  std::stringstream ss;
  ss << "miopenConvSolution_t time=" << algorithm.time
     << " workspace_size=" << algorithm.workspace_size
     << " solution_id=" << algorithm.solution_id
     << " algorithm=" << PrettyString(algorithm.algorithm);
  return ss.str();
}

namespace detail {

void check(miopenStatus_t expr, const char* file, int line, const char* cmd) {
  std::stringstream ess;
  ess << file << ':' << line << " MiOpen error=" << PrettyString(expr) << " ("
      << static_cast<int>(expr) << ") on " << cmd;
  if (expr != CL_SUCCESS) {
    throw std::runtime_error(ess.str());
  }
}

} // namespace detail
} // namespace miopen
} // namespace fl
