
#include "flashlight/fl/tensor/TensorExtension.h"

namespace fl {

class AutogradExtension : public TensorExtension {
 public:
  virutal Tensor conv2d(
      const Tensor& input,
      const Tensor& weights,
      const Tensor& bias,
      int sx = 1,
      int sy = 1,
      int px = 0,
      int py = 0,
      int dx = 1,
      int dy = 1,
      int groups = 1,
      std::shared_ptr<detail::ConvBenchmarks> benchmarks = nullptr) = 0;
};

} // namespace fl
