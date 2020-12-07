#pragma once

#include "flashlight/fl/nn/modules/Module.h"
#include "flashlight/fl/nn/nn.h"

namespace fl {
namespace app {
namespace objdet {

class PositionalEmbeddingSine : public Container {
  public:
    explicit PositionalEmbeddingSine(
        const int numPosFeats,
        const int temperature,
        const bool normalize,
        const float scale);

  std::vector<Variable> forward(const std::vector<Variable>& input) override;

  std::vector<Variable> operator()(const std::vector<Variable>& input);

  std::string prettyString() const override;

private:
 FL_SAVE_LOAD_WITH_BASE(fl::Container, numPosFeats_, temperature_, normalize_, scale_)
  int numPosFeats_;
  int temperature_;
  bool normalize_;
  float scale_;
    PositionalEmbeddingSine();
};

} // end namespace objdet
} // end namespace app
} // end namespace fl
CEREAL_REGISTER_TYPE(fl::app::objdet::PositionalEmbeddingSine)
