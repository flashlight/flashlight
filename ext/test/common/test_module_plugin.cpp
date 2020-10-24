#include <flashlight/flashlight/flashlight.h>

extern "C" fl::Module* createModule(int64_t nFeature, int64_t nLabel) {
  auto seq = new fl::Sequential();
  seq->add(std::make_shared<fl::Linear>(nFeature, nLabel));
  return seq;
}
