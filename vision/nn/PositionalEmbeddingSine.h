#pragma once

#include "flashlight/nn/modules/Module.h"

namespace fl {
namespace cv {

class PositionalEmbeddingSine : public UnaryModule {
  public:
    PositionalEmbeddingSine(
        const int numPosFeats,
        const int temperature,
        const bool normalize,
        const float scale);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;

private:
  const int numPosFeats_;
  const int temperature_;
  const bool normalize_;
  const float scale_;
};

} // end namespace fl
} // end namespace cv
