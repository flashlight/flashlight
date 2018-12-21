/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "View.h"

#include <flashlight/autograd/Functions.h>
#include <flashlight/common/Exception.h>
#include <flashlight/nn/Init.h>

namespace fl {

View::View(af::dim4 dims) :dims_(dims) {}

Variable View::forward(const Variable& input) {
  af::dim4 dims =dims_;

  return moddims(input, dims);
}

std::string View::prettyString() const {
  std::ostringstream ss;
  ss << "View (" <<dims_ << ")";
  return ss.str();
}

} // namespace fl
