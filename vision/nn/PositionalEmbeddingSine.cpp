#include "PositionalEmbeddingSine.h"

namespace fl {
namespace cv {

std::string PositionalEmbeddingSine::prettyString() const {
  return "PositionalEmbeddingSine";
}

PositionalEmbeddingSine::PositionalEmbeddingSine(
    const int numPosFeats,
    const int temperature,
    const bool normalize,
    const float scale) : 
  numPosFeats_(numPosFeats),
  temperature_(temperature),
  normalize_(normalize),
  scale_(scale) {
};

Variable PositionalEmbeddingSine::forward(const Variable& input) {
  auto nonMask = af::constant(1, { input.dims(0), input.dims(1), input.dims(3) });
  //auto nonMask = ~mask;

  af::array xEmbed = af::moddims(af::scan(nonMask, 0), 
      { 1, nonMask.dims(0), nonMask.dims(1), nonMask.dims(2) });
  af::array yEmbed = af::moddims(af::scan(nonMask, 1), 
      { 1, nonMask.dims(0), nonMask.dims(1), nonMask.dims(2) });
  auto dim = af::range(af::dim4(numPosFeats_), 0, f32);
  dim = af::pow(temperature_, ((2 * af::floor(dim / 2)) / numPosFeats_));
  auto posX = af::batchFunc(xEmbed, dim, af::operator/);
  auto posY = af::batchFunc(yEmbed, dim, af::operator/);

  auto posXSin = sin(posX(af::seq(0, af::end, 2), af::span, af::span, af::span));
  auto posXCos = cos(posX(af::seq(1, af::end, 2), af::span, af::span, af::span));
  auto posYSin = sin(posY(af::seq(0, af::end, 2), af::span, af::span, af::span));
  auto posYCos = cos(posY(af::seq(1, af::end, 2), af::span, af::span, af::span));

  auto interleave = [](af::array x, af::array y) {
    auto dims = x.dims();
    x = af::flat(x);
    y = af::flat(y);
    x = af::moddims(x, {1, x.dims(0) });
    y = af::moddims(y, {1, y.dims(0) });
    auto joined =  af::join(0, x, y);
    dims[0] = dims[0] * 2;
    return af::moddims(joined, dims);
  };

  posX = interleave(posXSin, posXCos);
  posY = interleave(posYSin, posYCos);
  af_print(posX);
  af_print(posY);
  auto result = af::join(0, posY, posX);
  af_print(result)
  result = af::reorder(result, 1, 2, 0, 3);
  return fl::Variable(result, false);
}

} // end namespace fl
} // end namespace cv
