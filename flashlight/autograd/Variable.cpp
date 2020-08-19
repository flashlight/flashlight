/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
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

#include "flashlight/autograd/Variable.h"

#include <af/internal.h>
#include <algorithm>
#include <cassert>
#include <functional>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "flashlight/common/CppBackports.h"

namespace fl {

Variable::Variable(af::array data, bool calcGrad) {
  sharedData_->data = std::move(data);
  sharedGrad_->calcGrad = calcGrad;
}

Variable::Variable(
    af::array data,
    std::vector<Variable> inputs,
    GradFunc gradFunc) {
  sharedData_->data = std::move(data);
  if (std::any_of(inputs.begin(), inputs.end(), [](const Variable& input) {
        return input.isCalcGrad();
      })) {
    sharedGrad_->calcGrad = true;
    sharedGrad_->inputs = std::move(inputs);
    sharedGrad_->gradFunc = std::move(gradFunc);
  }
}

Variable Variable::operator()(
    const af::index& s0,
    const af::index& s1, /* af::span */
    const af::index& s2, /* af::span */
    const af::index& s3 /* af::span */) const {
  auto result = array()(s0, s1, s2, s3);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [s0, s1, s2, s3, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    if (!inputs[0].isGradAvailable()) {
      auto grad = af::constant(0.0, inDims, inType);
      inputs[0].addGrad(Variable(grad, false));
    }
    auto& grad = inputs[0].grad().array();
    grad(s0, s1, s2, s3) += gradOutput.array();
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

af::array& Variable::array() const {
  return sharedData_->data;
}

Variable& Variable::grad() const {
  if (!sharedGrad_->calcGrad) {
    throw std::logic_error("gradient calculation disabled for this Variable");
  }

  if (!sharedGrad_->grad) {
    throw std::logic_error("gradient not calculated yet for this Variable");
  }

  return *sharedGrad_->grad;
}

std::vector<Variable>& Variable::getInputs() const {
  return sharedGrad_->inputs;
}

bool Variable::isCalcGrad() const {
  return sharedGrad_->calcGrad;
}

bool Variable::isGradAvailable() const {
  if (!sharedGrad_->calcGrad) {
    return false;
  }
  return sharedGrad_->grad != nullptr;
}

af::dim4 Variable::dims() const {
  return array().dims();
}

bool Variable::isempty() const {
  return array().isempty();
}

bool Variable::isLinear() const {
  return af::isLinear(array());
}

Variable Variable::linear() const {
  if (!isempty() && !isLinear()) {
    auto linearArray = af::array(dims(), type());
    af::copy(linearArray, array(), af::span);
    array() = linearArray;
  }
  return *this;
}

af::dtype Variable::type() const {
  return array().type();
}

dim_t Variable::elements() const {
  return array().elements();
}

size_t Variable::bytes() const {
  return array().bytes();
}

unsigned Variable::numdims() const {
  return array().numdims();
}

dim_t Variable::dims(unsigned dim) const {
  return array().dims(dim);
}

void Variable::eval() const {
  return array().eval();
}

void Variable::zeroGrad() {
  sharedGrad_->grad.reset();
}

void Variable::setCalcGrad(bool calcGrad) {
  sharedGrad_->calcGrad = calcGrad;
  if (!calcGrad) {
    sharedGrad_->gradFunc = nullptr;
    sharedGrad_->inputs.clear();
    sharedGrad_->grad.reset();
  }
}

void Variable::addGrad(const Variable& childGrad) {
  if (sharedGrad_->calcGrad) {
    if (sharedGrad_->grad) {
      // Prevent increment of array refcount to avoid a copy
      // if getting a device pointer. See
      // https://git.io/fp9oM for more
      sharedGrad_->grad = cpp::make_unique<Variable>(
          sharedGrad_->grad->array() + childGrad.array(), false);
      // Eval the JIT as a temporary workaround for
      // https://github.com/arrayfire/arrayfire/issues/2281
      sharedGrad_->grad->eval();
    } else {
      // Copy the childGrad Variable so as to share a reference
      // to the underlying childGrad.array() rather than copying
      // the array into a new variable
      sharedGrad_->grad = cpp::make_unique<Variable>(childGrad);
    }
  }
}

void Variable::registerGradHook(const GradHook& hook) {
  sharedGrad_->onGradAvailable = hook;
}

void Variable::clearGradHook() {
  sharedGrad_->onGradAvailable = nullptr;
}

void Variable::applyGradHook() {
  if (sharedGrad_->onGradAvailable) {
    assert(sharedGrad_->grad);
    sharedGrad_->onGradAvailable(*sharedGrad_->grad);
  }
}

void Variable::calcGradInputs(bool retainGraph) {
  if (sharedGrad_->gradFunc) {
    if (!sharedGrad_->grad) {
      throw std::logic_error("gradient was not propagated to this Variable");
    }

    sharedGrad_->gradFunc(sharedGrad_->inputs, *sharedGrad_->grad);
  }
  if (!retainGraph) {
    sharedGrad_->inputs.clear();
  }
}

void Variable::backward(const Variable& grad, bool retainGraph) {
  addGrad(grad);
  auto dag = build();
  for (auto iter = dag.rbegin(); iter != dag.rend(); iter++) {
    iter->calcGradInputs(retainGraph);
    iter->applyGradHook();
    if (!retainGraph) {
      *iter = Variable();
    }
  }
}

void Variable::backward(bool retainGraph) {
  auto ones = Variable(af::constant(1, dims()), false);
  backward(ones, retainGraph);
}

Variable Variable::col(int index) const {
  auto result = array().col(index);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [index, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().col(index) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::cols(int first, int last) const {
  auto result = array().cols(first, last);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [first, last, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().cols(first, last) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::row(int index) const {
  auto result = array().row(index);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [index, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().row(index) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::rows(int first, int last) const {
  auto result = array().rows(first, last);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [first, last, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().rows(first, last) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::slice(int index) const {
  auto result = array().slice(index);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [index, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().slice(index) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::slices(int first, int last) const {
  auto result = array().slices(first, last);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [first, last, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().slices(first, last) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::withoutData() const {
  Variable other;
  other.sharedGrad_ = sharedGrad_;
  return other;
}

Variable::DAG Variable::build() const {
  std::unordered_set<SharedGrad*> cache;
  DAG dag;
  std::function<void(const Variable&)> recurse;

  // Topological sort
  recurse = [&](const Variable& var) {
    auto id = var.sharedGrad_.get();
    if (cache.find(id) != cache.end()) {
      return;
    }
    for (const auto& input : var.getInputs()) {
      recurse(input);
    }
    cache.insert(id);
    dag.push_back(var);
  };

  recurse(*this);
  return dag;
}

} // namespace fl
