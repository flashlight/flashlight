/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * Applies a transformation on the input specified by a lambda function. For
 * example to add a \f$ 1 + log(x) \f$ layer to a container:
 * \code
    model.add(
      Transform([](const Variable& in) {
        return 1 + afnet::log(in);
      }
    );
    \endcode
 * Note this module cannot be serialized.
 */
class FL_API Transform : public UnaryModule {
 private:
  Transform() = default; // Intentionally private

  std::function<Variable(const Variable&)> func_;

  std::string name_;

  /**
   * Transform layers cannot be serialized. This function throws a runtime
   * exception.
   */
  FL_SAVE_LOAD_DECLARE()

 public:
  /**
   * Construct a Transform (lambda) layer.
   * @param func a lambda function which accepts an input Variable and returns
   * an output Variable.
   * @param name an optional name used by prettyString.
   */
  explicit Transform(
      const std::function<Variable(const Variable&)>& func,
      const std::string& name = "");

  Variable forward(const Variable& input) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

template <class Archive>
void Transform::save(Archive& /* ar */, const uint32_t /* version */) const {
  throw std::runtime_error("Transform module does not support serialization");
}

template <class Archive>
void Transform::load(Archive& /* ar */, const uint32_t /* version */) {
  throw std::runtime_error("Transform module does not support serialization");
}

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Transform)
CEREAL_REGISTER_POLYMORPHIC_RELATION(fl::Module, fl::Transform)
