/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <miopen/miopen.h>

#define MIOPEN_CHECK_ERR(expr) \
  ::fl::miopen::detail::check((expr), __FILE__, __LINE__, #expr)

namespace fl {
namespace miopen {

const void* kOne(const af::dtype t);
const void* kZero(const af::dtype t);

std::string PrettyString(miopenConvSolution_t algorithm);
std::string PrettyString(miopenConvAlgoPerf_t algorithm);
std::string PrettyString(miopenConvAlgorithm_t algorithm);
std::string PrettyString(miopenConvBwdDataAlgorithm_t algorithm);
std::string PrettyString(miopenConvBwdWeightsAlgorithm_t algorithm);
std::string PrettyString(miopenConvFwdAlgorithm_t algorithm);
std::string PrettyString(miopenStatus_t status);

namespace detail {

void check(miopenStatus_t err, const char* file, int line, const char* cmd);

} // namespace detail
} // namespace miopen
} // namespace fl
