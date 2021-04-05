/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#pragma once

#include <memory>
#include <stdexcept>
#include <unordered_map>

#include <cereal/types/tuple.hpp>
#include <cereal/types/unordered_map.hpp>

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

typedef std::shared_ptr<Module> ModulePtr;

/**
 * A computation unit capable of forward computation that contains a
 * collection of multiple `Module` and their respective parameters.
 */
class Container : public Module {
 private:
  // Keep track of location of submodule parameters in a map
  // from param index -> {module index, module param index}
  std::unordered_map<int, std::tuple<int, int>> childParamIdx_;

  FL_SAVE_LOAD_WITH_BASE(Module, modules_, childParamIdx_)

 protected:
  /**
   * A collection of modules contained within a `Container`.
   */
  std::vector<ModulePtr> modules_;

  Container();

 public:
  /**
   * Adds a module to a `Container` by making a copy of the underlying module.
   * Note that parameters are still shared, due to Variable's copy semantics.
   *
   * @param module the module to add.
   */
  template <typename T>
  void add(const T& module) {
    add(std::make_shared<T>(module));
  }

  /**
   * Adds a module to `modules_`, and adds parameters to the container's
   * `params_`.
   *
   * @param module the module to add.
   */
  template <typename T>
  void add(std::shared_ptr<T> module) {
    if (!module) {
      throw std::invalid_argument("can't add null Module to Container");
    }
    modules_.emplace_back(module);
    for (int i = 0; i < module->params().size(); i++) {
      childParamIdx_[params_.size()] = std::make_tuple(modules_.size() - 1, i);
      params_.push_back(module->param(i));
    }
  }

  /**
   * Returns a pointer to the module at the specified index in the container's
   * `modules_`.
   *
   * @param id the index of the module to return
   * @return a pointer to the requested module
   */
  ModulePtr module(int id) const;

  /**
   * Returns pointers to each of `Module` in the `Container`.
   *
   * @return an ordered vector of pointers for each module.
   */
  std::vector<ModulePtr> modules() const;

  /**
   * Switches all modules in the `Container` into train mode. See `Module`.
   */
  void train() override;

  /**
   * Switches all modules in the `Container` into eval mode. See `Module`.
   */
  void eval() override;

  /**
   * Sets a parameter at a specified position with a new, given one.
   *
   * If the specified position is not valid (it is negative or greater than
   * ``params_.size() - 1``), then an error will be thrown. A new parameter
   * will not be created at a specified index if out of bounds.
   *
   * @param var the new replacement `Variable`
   * @param position The index of the parameter which will be replaced in
   * `params_`
   */
  void setParams(const Variable& var, int position) override;

  /**
   * Generates a stringified representation of the module.
   *
   * @return a string containing the module label
   */
  virtual std::string prettyString() const override;
};

/**
 * A `Container` representing an ordered sequence of modules, which is capable
 * of forward computation through each of its modules, in order.
 *
 * Usage:
 * \code
   Sequential mySequential();
   // Assume we've defined implemented three modules, mod1, mod2, mod3
   mySequential.add(mod1);
   mySequential.add(mod2);
   mySequential.add(mod3);
   // Performing forward computation will forward through each `Module` in order
   auto result = mySequential.forward(myInput);
   // We can also inspect internal state
   assert(mySequential.modules().size() == 3); // true
   assert(
       mod1.params().size() + mod2.params.size() +
           mod3.params().size() ==
       mySequential.params().size()); // true
   \endcode
 */
class Sequential : public Container {
 public:
  Sequential();

  /**
   * Performs forward computation for the `Sequential`, calling `forward`, in
   * order, for each `Module`, and feeding the result as input to the next
   * `Module`.
   *
   * @param input the value on which the `Container` will perform forward
   * computation.
   * @return a `Variable` tensor containing the result of the forward
   * computation
   */
  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  Variable forward(const Variable& input);

  Variable operator()(const Variable& input);

  /**
   * Generates a stringified representation of the `Sequential` by concatenating
   * string representations for each contained `Module`
   *
   * @return a string containing the module label
   */
  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Container)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Container)
CEREAL_REGISTER_TYPE(fl::Sequential)
