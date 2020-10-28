#include <flashlight/nn/modules/PrecisionCast.h>

namespace fl {

PrecisionCast::PrecisionCast(af::dtype targetType) : targetType_(targetType) {}

std::vector<Variable> PrecisionCast::forward(
    const std::vector<Variable>& inputs) {
  std::vector<Variable> outputs;
  for (auto input : inputs) {
    auto output = input.as(targetType_);
    outputs.push_back(output);
  }
  return outputs;
}

Variable PrecisionCast::forward(const Variable& input) {
  return forward(std::vector<Variable>{input}).front();
}

Variable PrecisionCast::operator()(const Variable& input) {
  return this->forward(input);
}

std::string PrecisionCast::prettyString() const {
  std::ostringstream ss;
  ss << "PrecisionCast";
  ss << " * -> " << afTypeToString(targetType_);
  return ss.str();
}

} // namespace fl
