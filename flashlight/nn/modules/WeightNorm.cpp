/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "Conv2D.h"
#include "Linear.h"
#include "WeightNorm.h"

namespace fl {

void WeightNorm::transformDims() {
  normDim_.clear();
  int v_numdims = params_[0].array().numdims();
  if (dim_ < 0 || dim_ > v_numdims) {
    throw std::invalid_argument("invalid dimension for WeightNorm");
  }
  for (int i = 0; i < v_numdims; i++) {
    if (i != dim_) {
      normDim_.push_back(i);
    }
  }
}

void WeightNorm::computeWeight() {
  auto v = params_[0];
  auto g = params_[1];
  auto wt = v * tileAs(g / norm(v, normDim_), v);
  modules_[0]->setParams(wt, 0);
}

void WeightNorm::setParams(const Variable& var, int position) {
  Module::setParams(var, position);
  // it is necessary to copy all params to the parent module
  // due to copies stored in the parent module (not pointers)
  if (position == 2) {
    modules_[0]->setParams(var, 1);
  } else if (position <= 1) {
    computeWeight();
  }
}

std::vector<Variable> WeightNorm::forward(const std::vector<Variable>& inputs) {
  if (train_) {
    computeWeight();
  }
  return modules_[0]->forward(inputs);
}

void WeightNorm::eval() {
  Container::eval();
  computeWeight();
}

std::string WeightNorm::prettyString() const {
  std::ostringstream ss;
  ss << "WeightNorm";
  ss << " (" << modules_[0]->prettyString() << ", " << dim_ << ")";
  return ss.str();
}

} // namespace fl
