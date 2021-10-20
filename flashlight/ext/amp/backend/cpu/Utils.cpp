
#include <arrayfire.h>

#include "flashlight/ext/amp/Utils.h"
#include "flashlight/fl/common/Utils.h"

namespace fl {
namespace ext {
void validityCheck(af::array& in, af::array& isInvalidArray) {
  if (fl::isInvalidArray(in)) {
    isInvalidArray = af::constant(1, 1, 1, 1, 1, s32);
  }
}

void scaleGrads(af::array& grads, af::array& scaleFactor, af::array& isInvalidArray) {
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

bool decreaseScaleFactor(
    af::array& scaleFactor,
    af::array& isInvalidArray,
    const af::array& minScaleFactor) {
  if (isInvalidArray.scalar<int>() == 1) {
    if (scaleFactor.scalar<float>() / 2.0f >= minScaleFactor.scalar<float>()) {
      scaleFactor = scaleFactor / 2.0f;
    }
    isInvalidArray = af::constant(0, 1, 1, 1, 1, s32);
    return false;
  }
  return true;
}

void increaseScaleFactor(
    af::array& scaleFactor,
    const af::array& maxScaleFactor,
    const ScaleFactorIncreaseForm& increaseForm) {
  if (increaseForm == ScaleFactorIncreaseForm::MULTIPLICATIVE) {
    if (scaleFactor.scalar<float>() * 2 <= maxScaleFactor.scalar<float>()) {
      scaleFactor = scaleFactor * 2;
    }
  } else {
    if (scaleFactor.scalar<float>() + 2 <= maxScaleFactor.scalar<float>()) {
      scaleFactor = scaleFactor + 2;
    }
  }
}

} // namespace ext
} // namespace fl
