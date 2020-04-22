#pragma once

#include <flashlight/autograd/Variable.h>
#include <flashlight/common/Serialization.h>
#include <flashlight/common/Utils.h>
#include <flashlight/nn/modules/Module.h>

namespace fl {
/**
 * Precision Cast Module. Casts the input from its original precision to the
 * target precision. Precision cast only alters the underlying array and leaves
 * other attributes of the input variable unchanged.
 */
class PrecisionCast : public Module {
  af::dtype targetType_;
  PrecisionCast() = default;
  FL_SAVE_LOAD_WITH_BASE(Module, targetType_)
 public:
  PrecisionCast(af::dtype targetType);
  std::vector<Variable> forward(const std::vector<Variable>& inputs) override;
  Variable forward(const Variable& input);
  Variable operator()(const Variable& input);
  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::PrecisionCast)