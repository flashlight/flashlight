/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>

#include "flashlight/fl/dataset/ResampleDataset.h"

namespace fl {

/**
 * A view into a dataset, with indices permuted randomly.
 *
 * Example:
  \code{.cpp}
  // Make a dataset with 100 samples
  auto tensor = fl::rand({5, 4, 100});
  std::vector<Tensor> fields{tensor};
  auto ds = std::make_shared<TensorDataset>(fields);

  // Shuffle it
  ShuffleDataset shuffleds(ds);
  std::cout << shuffleds.size() << "\n"; // 100
  std::cout << "first try" << shuffleds.get(0)["x"] << std::endl;

  // Reshuffle it
  shuffleds.resample();
  std::cout << "second try" << shuffleds.get(0)["x"] << std::endl;
  \endcode
 */
class FL_API ShuffleDataset : public ResampleDataset {
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
