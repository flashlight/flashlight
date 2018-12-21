/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "Container.h"

#include <flashlight/autograd/Variable.h>

namespace fl {

Container::Container() = default;

ModulePtr Container::module(int id) {
  return modules_[id];
}

std::vector<ModulePtr> Container::modules() {
  return modules_;
}

void Container::train() {
  Module::train();
  for (auto& module : modules_) {
    module->train();
  }
}

void Container::eval() {
  Module::eval();
  for (auto& module : modules_) {
    module->eval();
  }
}

void Container::setParams(const Variable& var, int position) {
  Module::setParams(var, position);
  auto indices = childParamIdx_.find(position);
  if (indices != childParamIdx_.end()) {
    int midx, pidx;
    std::tie(midx, pidx) = indices->second;
    modules_[midx]->setParams(var, pidx);
  }
}

Sequential::Sequential() = default;

Variable Sequential::forward(const Variable& input) {
  Variable output = input;
  for (auto& module : modules_) {
    output = module->forward(output);
  }
  return output;
}

std::string Sequential::prettyString() const {
  std::ostringstream ss;
  ss << "Sequential";
  ss << " [input";
  for (int i = 0; i < modules_.size(); ++i) {
    ss << " -> (" << i << ")";
  }
  ss << " -> output]";
  for (int i = 0; i < modules_.size(); ++i) {
    ss << "\n\t(" << i << "): " << modules_[i]->prettyString();
  }
  return ss.str();
}

} // namespace fl
