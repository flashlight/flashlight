#include <arrayfire.h>

namespace fl {
af::array relativePositionalEmbeddingRotate(const af::array& input) {
  int d0 = input.dims(0);
  int d1 = input.dims(1);
  int d2 = input.dims(2);
  int d3 = input.dims(3);
  auto data =
      af::join(0, input, af::constant(0.0, d1, d1, d2, d3, input.type()));
  data = af::moddims(data, af::dim4((d0 + d1) * d1, 1, d2, d3));
  data = data.rows(0, (d1 + d0 - 1) * d1 - 1);
  data = af::moddims(data, af::dim4(d0 + d1 - 1, d1, d2, d3));
  return data;
}
} // namespace fl
