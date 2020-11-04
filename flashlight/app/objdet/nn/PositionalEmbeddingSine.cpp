#include "PositionalEmbeddingSine.h"
#include <cassert>

namespace fl {
namespace app {
namespace object_detection {

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
  auto inputDims = input.dims();
  // Input mask will be [ w x h x 1 x b ]
  // but implemention expects [ w x h x b ] in order to do interleaves easier
  auto nonMask = af::moddims(input.array(), { inputDims[0], inputDims[1], inputDims[3], 1 });

  //auto nonMask = ~mask;
  //
  auto expandDims = [](const af::array in) {
    auto dims = in.dims();
    assert(dims[3] == 1);
    return af::moddims(in, { 1, dims[0], dims[1], dims[2] });
  };

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

  af::array xEmbed = af::scan(nonMask, 0);
  af::array yEmbed = af::scan(nonMask, 1);

  auto dim = af::range(af::dim4(numPosFeats_), 0, f32);
  dim = af::pow(temperature_, ((2 * af::floor(dim / 2)) / numPosFeats_));

  auto posX = af::batchFunc(expandDims(xEmbed), dim, af::operator/);
  auto posY = af::batchFunc(expandDims(yEmbed), dim, af::operator/);

  auto posXSin = af::sin(posX(af::seq(0, af::end, 2), af::span));
  auto posXCos = af::cos(posX(af::seq(1, af::end, 2), af::span));
  auto posYSin = af::sin(posY(af::seq(0, af::end, 2), af::span));
  auto posYCos = af::cos(posY(af::seq(1, af::end, 2), af::span));


  posX = interleave(posXSin, posXCos);
  posY = interleave(posYSin, posYCos);
  auto result = af::join(0, posY, posX);
  result = af::reorder(result, 1, 2, 0, 3);
  return fl::Variable(result, false);
}

} // end namespace image
} // end namespace ext
} // end namespace fl
