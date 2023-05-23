/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

  /**
   * Removes all modules from the container.
   */
  virtual void clear();

  /**
   * Find orphaned params (i.e. params not in modules contained in the modules_
   * list).
   * @return A multimap of orphaned params and the module index they appear
   * after
   */
  std::unordered_multimap<int, int> getOrphanedParamsIdxMap() const;

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
 * An extension of `Container` which enables cloning of the derrived container,
 * performing a deep copy of the modules (and parameters) of the container.
 * **Note** If Derived implements members which use shallow copy semantics like
 * shared_ptr (even if they are added to `Container::modules_`), Derived must
 * implement its own copy function to handle those members and/or parameters.
 * @tparam Derived The class which is inheriting from `CloneableContainer`
 */
template <typename Derived>
class CloneableContainer : public Container {
 public:
  /**
   * Copies the container via copy constructor and performs a deep copy of the
   * modules of this container to the new container.
   * @return A copy of this object
   */
  Derived copy() const {
    Derived deepCopy(static_cast<const Derived&>(*this));
    deepCopy.clear();
    deepCopy.params_.reserve(Module::params_.size());
    deepCopy.modules_.reserve(Container::modules_.size());

    auto orphanParamIdxMap = getOrphanedParamsIdxMap();
    for (int i = -1; i < static_cast<int>(Container::modules_.size()); ++i) {
      if (i >= 0) {
        const auto& mod = Container::modules_[i];
        deepCopy.add(mod->clone());
      }
      auto [paramIter, pEnd] = orphanParamIdxMap.equal_range(i);
      for (; paramIter != pEnd; ++paramIter) {
        const auto& param = Module::params_[paramIter->second];
        deepCopy.params_.emplace_back(
            param.tensor().copy(), param.isCalcGrad());
      }
    }

    return deepCopy;
  }

  /**
   * Clones the `Container`, performing a deep copy. See `copy()` for more
   * details.
   * @return A shared pointer to the base class of the cloned module.
   */
  std::shared_ptr<Module> clone() const override {
    return std::make_shared<Derived>(copy());
  }
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
class Sequential : public CloneableContainer<Sequential> {
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
