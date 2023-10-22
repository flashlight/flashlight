/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/Serialization.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace fl {

/**
 * An abstract computation unit capable of forward computation. Also
 * contains a collection of parameters that can be mutated, and will be
 * serialized and deserialized with the module.
 */
class FL_API Module {
 private:
  /**
   * Serialize the module's parameters.
   */
  FL_SAVE_LOAD(params_, train_)

 protected:
  /**
   * Parameters of module, represented as a collection of `Variable`, whose
   * ordering is based on the implementation of the respective module.
   */
  std::vector<Variable> params_;

  /**
   * A flag specifying whether or not the module is in `train` mode. If
   * `Module::train()` is called, it will be set to true, and if
   * `Module::eval()` is called, it will be set to false.
   */
  bool train_ = true;

  /**
   * An empty module constructor, which creates a module with no parameters.
   *
   */
  Module();

  /**
   * Constructs a module given its parameters.
   *
   * @param params a vector of `Variable` which will replace `params_`
   * This changes all parameters so that gradient calculation will be
   * enabled/disabled for any calls to `forward`.
   */
  explicit Module(const std::vector<Variable>& params);

 public:
  /**
   * Gets the parameters of the module.
   *
   * @return the modules parameters as a vector of `Variable`
   */
  std::vector<Variable> params() const;

  /**
   * Gets the nunber of parameter tensors of the module.
   *
   * @return the number of parameter tensors
   */
  int numParamTensors() const;

  /**
   * Switches the module to training mode. Changes all parameters so that
   * gradient calculation will be enabled for any calls to `forward`.
   */
  virtual void train();

  /**
   * Switches the module to evaluation mode. Changes all parameters so that
   * gradient calculation will be disabled for any calls to `forward`.
   */
  virtual void eval();

  /**
   * Returns a module parameter given a particular position.
   *
   * @param position the index of the requested parameter in `params_`
   * @return a `Variable` tensor for the parameter at the requested position
   */
  Variable param(int position) const;

  /**
   * Sets a parameter at a specified position with a new, given one.
   *
   * If the specified position is not valid (it is negative or greater than
   * ``params_.size() - 1``), then an error will be thrown. A new parameter
   * will not be created at a specified index if out of bounds.
   *
   * @param[in] var the new replacement `Variable`
   * @param position The index of the parameter which will be replaced in
   * `params_`
   */
  virtual void setParams(const Variable& var, int position);

  /**
   * Copies the modules parameters, detaching from the computation graph.
   *
   * @return a copy of the modules parameters as a vector of `Variable`
   */
  virtual std::vector<Variable> copyParams() const;

  /**
   * Clears references to gradient Variables for all parameters in the module.
   */
  void zeroGrad();

  /**
   * Performs forward computation for the module, given some inputs.
   *
   * @param inputs the values to compute forward computation for the
   * module.
   * @return a vector of `Variable` tensors containing the result of
   * the forward computation
   */
  virtual std::vector<Variable> forward(
      const std::vector<Variable>& inputs) = 0;

  /**
   * Overload for forward computation for the module.
   *
   * @param inputs the values to compute forward computation for the
   * module.
   * @return a vector of `Variable` tensors containing the result of
   * the forward computation
   */
  std::vector<Variable> operator()(const std::vector<Variable>& inputs);

  /**
   * Clone the module via deep copy of its parameters and members.
   *
   * @return A unique pointer of the cloned module.
   */
  virtual std::unique_ptr<Module> clone() const = 0;

  /**
   * Generates a stringified representation of the module.
   *
   * @return a string containing the module label
   */
  virtual std::string prettyString() const = 0;

  virtual ~Module() = default;
};

/**
 * An extension of `Module` which supports only forward computation on a single
 * `Variable` with a single `Variable` as output.
 * For example, `Sigmoid` module can be derived from `UnaryModule`.
 */
class FL_API UnaryModule : public Module {
 public:
  UnaryModule();

  explicit UnaryModule(const std::vector<Variable>& params);

  std::vector<Variable> forward(const std::vector<Variable>& inputs) override;

  virtual Variable forward(const Variable& input) = 0;

  Variable operator()(const Variable& input);

  virtual ~UnaryModule() = default;

 private:
  FL_SAVE_LOAD_WITH_BASE(Module)
};

/**
 * An extension of `Module` which supports only forward computation on a pair of
 * `Variable`s with a single `Variable` as output.
 * For example, `BinaryCrossEntropy` Loss can be derived from `BinaryModule`.
 */
class FL_API BinaryModule : public Module {
 public:
  BinaryModule();

  explicit BinaryModule(const std::vector<Variable>& params);

  std::vector<Variable> forward(const std::vector<Variable>& inputs) override;

  virtual Variable forward(const Variable& input1, const Variable& input2) = 0;

  Variable operator()(const Variable& input1, const Variable& input2);

  virtual ~BinaryModule() = default;

 private:
  FL_SAVE_LOAD_WITH_BASE(Module)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::UnaryModule)
CEREAL_REGISTER_TYPE(fl::BinaryModule)
