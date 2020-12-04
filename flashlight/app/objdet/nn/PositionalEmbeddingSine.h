#pragma once

#include "flashlight/fl/nn/modules/Module.h"

namespace fl {
namespace app {
namespace objdet {

class PositionalEmbeddingSine : public UnaryModule {
  public:
    PositionalEmbeddingSine(
        const int numPosFeats=64,
        const int temperature=10000,
        const bool normalize=false,
        const float scale=0.0f);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;

private:
  const int numPosFeats_;
  const int temperature_;
  const bool normalize_;
  const float scale_;
};

} // end namespace objdet
} // end namespace app
} // end namespace fl
