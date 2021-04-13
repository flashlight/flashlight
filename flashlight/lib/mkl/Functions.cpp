/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/lib/mkl/Functions.h"

#include <sstream>

#include <mkl.h>

namespace fl {
namespace lib {
namespace mkl {

#define FL_VSL_CHECK(cmd) \
  ::fl::lib::mkl::vslCheck(cmd, __FILE__, __LINE__, #cmd)

void vslCheck(MKL_INT err, const char* file, int line, const char* cmd) {
  if (err != VSL_STATUS_OK) {
    std::ostringstream ess;
    ess << file << ':' << line << "] MKL-VSL error: " << err << " cmd:" << cmd;
    throw std::runtime_error(ess.str());
  }
}

std::vector<float> Correlate(
    const std::vector<float>& kernel,
    const std::vector<float>& input) {
  std::vector<float> output(kernel.size() + input.size() - 1, 0);
  VSLConvTaskPtr task;
  FL_VSL_CHECK(vslsConvNewTask1D(
      &task, VSL_CONV_MODE_AUTO, kernel.size(), input.size(), output.size()));
  FL_VSL_CHECK(vslsConvExec1D(
      task, kernel.data(), 1, input.data(), 1, output.data(), 1));
  FL_VSL_CHECK(vslConvDeleteTask(&task));
  return output;
}

} // namespace mkl
} // namespace lib
} // namespace fl
