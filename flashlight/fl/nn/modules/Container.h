/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include <cereal/types/tuple.hpp>
#include <cereal/types/unordered_map.hpp>

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

typedef std::shared_ptr<Module> ModulePtr;

/**
 * A computation unit capable of forward computation that contains a
 * collection of multiple `Module` and their respective parameters.
 */
class FL_API Container : public Module {
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
   * Removes all modules and parameters from the container.
   */
  virtual void clear();

  /**
   * Find orphaned params (i.e. params not in modules contained in the modules_
   * list). This can be used to preserve the order of orphaned params when
   * copying/cloning a container. std::unordered_multimap<module_idx, param_idx>
   * The module_idx is used to identify after which module params should be
   * serted and the param_idx is used to index the specific param. The following
   * example demonstrates its usage by ensuring params and modules are inserted
   * in the same order when making a copy:
   * \code
      void copy(const MyContainer& other) {
        auto orphanParamIdxMap = other.getOrphanedParamsIdxMap();
        for (int i = -1; i < static_cast<int>(other.modules_.size()); ++i) {
          if (i >= 0) {
            add(other.modules_[i]->clone());
          }
          auto [paramIter, pEnd] = orphanParamIdxMap.equal_range(i);
          for (; paramIter != pEnd; ++paramIter) {
            const auto& param = other.params_[paramIter->second];
            params_.emplace_back(param.copy());
          }
        }
      }
     \endcode
   *
   * A module_idx of -1 indicates the orphaned params are to be inserted
   * before the first module
   *
   * @return A multimap of orphaned params and the module index they appear
   * after
   */
  std::unordered_multimap<int, int> getOrphanedParamsIdxMap() const;

 public:
  /**
   * Adds a module to a `Container` by making a copy of the underlying module if
   * an lvalue or moving it if and rvalue
   *
   * @param[in] module the module to add.
   */
  template <typename T>
  void add(T&& module) {
    static_assert(
        !std::is_lvalue_reference_v<T>,
        "add() can only accept rvalues. Use std::move().");
    add(std::make_shared<std::decay_t<T>>(std::forward<T>(module)));
  }

  /**
   * Adds a module to a `Container` by moving it and taking ownership.
   *
   * @param module the module to add.
   */
  template <typename T>
  void add(std::unique_ptr<T> module) {
    add(std::shared_ptr<T>(std::move(module)));
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
    for (int i = 0; i < module->numParamTensors(); i++) {
      childParamIdx_[params_.size()] = std::make_tuple(modules_.size(), i);
      params_.push_back(module->param(i));
    }
    modules_.emplace_back(std::move(module));
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
 * Adds a copy constructor, copy assignment operator, move constructor, move
 * assignment operator and clone method to the class. This should only be used
 * if the class basically acts as a wrapper around a container such that no
 * custom module ownership is used. Users should implement these methods
 * themselves if any custom ownership is utilised.
 * The following is an example of custom ownership, where a shared_ptr is used
 * to share ownership between the base Container class an MyContainer class
 * such that they need to be manually syncronised when performing a copy.
 * \code
    class MyContainer : public Container {
    public:
      MyContainer() {
        lin_ = std::make_shared<Linear>(10, 20);
        add(lin_);
      }
      std::shared_ptr<Linear> lin_;
    };
   \endcode
 */
#define FL_BASIC_CONTAINER_CLONING(ContainerClass)             \
  ContainerClass(const ContainerClass& other) {                \
    train_ = other.train_;                                     \
    for (auto& mod : other.modules_) {                         \
      add(mod->clone());                                       \
    }                                                          \
  }                                                            \
  ContainerClass& operator=(const ContainerClass& other) {     \
    train_ = other.train_;                                     \
    clear();                                                   \
    for (auto& mod : other.modules_) {                         \
      add(mod->clone());                                       \
    }                                                          \
    return *this;                                              \
  }                                                            \
  ContainerClass(ContainerClass&& other) = default;            \
  ContainerClass& operator=(ContainerClass&& other) = default; \
  std::unique_ptr<Module> clone() const override {             \
    return std::make_unique<ContainerClass>(*this);            \
  }

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
class FL_API Sequential : public Container {
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

  FL_BASIC_CONTAINER_CLONING(Sequential)

 private:
  FL_SAVE_LOAD_WITH_BASE(Container)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Container)
CEREAL_REGISTER_TYPE(fl::Sequential)
