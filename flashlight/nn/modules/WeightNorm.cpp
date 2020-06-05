/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>

#include "flashlight/nn/modules/Conv2D.h"
#include "flashlight/nn/modules/Linear.h"
#include "flashlight/nn/modules/WeightNorm.h"

namespace fl {

void WeightNorm::transformDims() {
  normDim_.clear();
  int vNumdims = module_->param(0).array().numdims();
  if (dim_ < 0 || dim_ > vNumdims) {
    throw std::invalid_argument("invalid dimension for WeightNorm");
  }
  for (int i = 0; i < vNumdims; i++) {
    if (i != dim_) {
      normDim_.push_back(i);
    }
  }
}

void WeightNorm::computeWeight() {
  auto v = params_[0];
  auto g = params_[1];
  auto wt = v * tileAs(g / norm(v, normDim_), v);
  module_->setParams(wt, 0);
}

void WeightNorm::setParams(const Variable& var, int position) {
  Module::setParams(var, position);
  // it is necessary to copy all params to the parent module
  // due to copies stored in the parent module (not pointers)
  if (position == 2) {
    module_->setParams(var, 1);
  } else if (position <= 1) {
    computeWeight();
  }
}

std::vector<Variable> WeightNorm::forward(const std::vector<Variable>& inputs) {
  if (train_) {
    computeWeight();
  }
  return module_->forward(inputs);
}

ModulePtr WeightNorm::module() const {
  return module_;
}

void WeightNorm::train() {
  Module::train();
  module_->train();
}

void WeightNorm::eval() {
  Module::eval();
  module_->eval();
  computeWeight();
}

std::string WeightNorm::prettyString() const {
  std::ostringstream ss;
  ss << "WeightNorm";
  ss << " (" << module_->prettyString() << ", " << dim_ << ")";
  return ss.str();
}

} // namespace fl
