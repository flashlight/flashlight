/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/dataset/ResampleDataset.h"

#include <random>

namespace fl {

/**
 * A view into a dataset, with indices permuted randomly.
 *
 * Example:
  \code{.cpp}
  // Make a dataset with 100 samples
  auto tensor = af::randu(5, 4, 100);
  std::vector<af::array> fields{tensor};
  auto ds = std::make_shared<TensorDataset>(fields);

  // Shuffle it
  ShuffleDataset shuffleds(ds);
  std::cout << shuffleds.size() << "\n"; // 100
  af::print("first try", shuffleds.get(0)["x"]);

  // Reshuffle it
  shuffleds.resample();
  af::print("second try", shuffleds.get(0)["x"]);
  \endcode
 */
class ShuffleDataset : public ResampleDataset {
 public:
  /**
   * Creates a `ShuffleDataset`.
   * @param[in] dataset The underlying dataset.
   * @param[seed] seed initial seed to be used.
   */
  explicit ShuffleDataset(std::shared_ptr<const Dataset> dataset, int seed = 0);

  /**
   * Generates a new random permutation for the dataset.
   */
  void resample();

  /**
   * Sets the PRNG seed.
   * @param[in] seed The desired seed.
   */
  void setSeed(int seed);

 protected:
  std::mt19937_64 rng_;
};

} // namespace fl
