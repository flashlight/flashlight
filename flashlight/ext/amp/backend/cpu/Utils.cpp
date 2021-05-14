
#include <arrayfire.h>

#include "flashlight/ext/amp/Utils.h"
#include "flashlight/fl/common/Utils.h"

namespace fl {
namespace ext {
void validityCheck(af::array& in, af::array& flag) {
  if (fl::isInvalidArray(in)) {
    flag = af::constant(1, 1, 1, 1, 1, s32);
  }
}

void scaleGrads(af::array& grads, af::array& scaleFactor, af::array& flag) {
  grads = grads / scaleFactor.scalar<float>();
}

fl::Variable scaleLoss(fl::Variable& loss, fl::Variable& scaleFactor) {
  float scaleFactorValue = scaleFactor.scalar<float>();
  auto scaledLoss = loss.array() * scaleFactorValue;
  auto gradFunc = [scaleFactorValue](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto grad =
        af::constant(scaleFactorValue, inputs[0].dims(), inputs[0].type());
    inputs[0].addGrad(fl::Variable(grad, false));
  };

  return fl::Variable(scaledLoss, {loss, scaleFactor}, gradFunc);
}

bool adjustScaleFactor(af::array& scaleFactor, af::array& flag) {
  if (flag.scalar<int>() == 1) {
    scaleFactor = scaleFactor / 2.0f;
    flag = af::constant(0, 1, 1, 1, 1, s32);
    return false;
  }
  return true;
}

} // namespace ext
} // namespace fl
