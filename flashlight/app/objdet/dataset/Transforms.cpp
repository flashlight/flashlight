#include "flashlight/app/objdet/dataset/BoxUtils.h"
#include "flashlight/ext/image/af/Transforms.h"

#include <assert.h>

namespace {

// TODO consolidate
af::array
crop(const af::array& in, const int x, const int y, const int w, const int h) {

  assert(x + w - 1 < in.dims(0));
  assert(y + h - 1 < in.dims(1));
  return in(af::seq(x, x + w - 1), af::seq(y, y + h - 1), af::span, af::span);
}
}


namespace fl {
namespace app {
namespace objdet {

std::vector<af::array> crop(
    const std::vector<af::array>& in,
    int x,
    int y,
    int tw,
    int th
    ) {
    const af::array& image = in[0];
    const af::array croppedImage = ::crop(image, x, y, tw, th);

    const af::array& boxes = in[3];

    const std::vector<int> translateVector = { x, y, x, y };
    const std::vector<int> maxSizeVector = { tw, th };

    const af::array translateArray = af::array(4, translateVector.data());
    const af::array maxSizeArray = af::array(2, maxSizeVector.data());

    af::array croppedBoxes = boxes;
    af::array labels = in[4];

    if(!croppedBoxes.isempty()) {
      croppedBoxes = af::batchFunc(croppedBoxes, translateArray, af::operator-);
      croppedBoxes = af::moddims(croppedBoxes, { 2, 2, boxes.dims(1)});
      croppedBoxes = af::batchFunc(croppedBoxes, maxSizeArray, af::min); 
      croppedBoxes = af::max(croppedBoxes, 0.0);
      af::array keep = allTrue(croppedBoxes(af::span, af::seq(1, 1)) > croppedBoxes(af::span, af::seq(0, 0)));
      croppedBoxes = af::moddims(croppedBoxes, { 4, boxes.dims(1) } );
      croppedBoxes = croppedBoxes(af::span, keep);
      labels  = labels(af::span, keep);
    }
    return { croppedImage, in[1], in[2], croppedBoxes, labels };
};

} // namespace objdet
} // namespace app
} // namespace fl
