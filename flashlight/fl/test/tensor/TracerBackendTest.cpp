/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <utility>

#include <gtest/gtest.h>

#include "flashlight/fl/tensor/DefaultTensorType.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/TensorAdapter.h"
#include "flashlight/fl/tensor/backend/trace/TensorTracer.h"
#include "flashlight/fl/tensor/backend/trace/TracerTensor.h"

// TODO: remove me
#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

using namespace fl;

TEST(TracerBackendTest, trace) {
  auto tracer =
      std::make_shared<TensorTracer>(std::make_unique<std::stringstream>());
  auto& tracerStream = dynamic_cast<std::stringstream&>(tracer->getStream());

  auto& tracerBackend = fl::TracerBackend<fl::ArrayFireBackend>::getInstance();
  tracerBackend.setTracer(tracer);

  // Tracing code
  auto a = fl::full({3, 3}, 6.);

  std::cout << "trace " << tracerStream.str() << std::endl;
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  // TODO: this isn't using DefaultTensorType_t -- add DefaultTensorBackend_t to
  // DefaultTensorType.h
  fl::setDefaultTensorType<fl::TracerTensor<
      fl::ArrayFireTensor,
      fl::TracerBackend<fl::ArrayFireBackend>>>();

  return RUN_ALL_TESTS();
}
