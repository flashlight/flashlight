/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cereal/types/set.hpp>
#include <cereal/types/unordered_set.hpp>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/nn/modules/Container.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * A module for creating a generic Residual block as given by [He et al
 (2015)](https://arxiv.org/abs/1512.03385) and [Kumar et al
 (2015)](https://arxiv.org/abs/1505.00387).
 *
 * Example:
  \code{.cpp}
  auto res = Residual();
  // Add multiple layers
  res.add(Conv2D(30, 50, 9, 7, 2, 3, 3, 2));
  res.add(BatchNorm(2, 50));
  res.add(ReLU());
  // Add a shortcut from the input to the block to the third layer
  res.addShortcut(0, 3);
  // Add a shortcut from the second layer to the output
  res.addShortcut(2, 4);
  // Scale the inputs to the third layer by some constant
  res.addScale(3, 0.5);

  // Create a model
  Sequential model;
  // ...
  // Add our residual block as needed
  model.add(res);
  model.add(Pool2D(2, 3, 1, 1, 1, 1, PoolingMode::MAX));
  model.add(res);
  // ...
  \endcode
 */
class FL_API Residual : public Container {
 private:
  FL_SAVE_LOAD_WITH_BASE(Container, shortcut_, scales_, projectionsIndices_)

  void checkShortcut(int fromLayer, int toLayer);
  void processShortcut(int fromLayer, int toLayer, int projectionIndex);
  Variable applyScale(const Variable& input, const int layerIndex);

  // Maps end -> start
  std::unordered_map<int, std::unordered_map<int, int>> shortcut_;
  // Indices of projection layers
  std::unordered_set<int> projectionsIndices_;
  std::unordered_map<int, float> scales_;

 public:
  Residual() = default;

  std::unordered_set<int> getProjectionsIndices() const;

  /**
   * Adds a scaling factor to all residual connections connecting to a layer
   * given by some index index. Given some scale \f$ \alpha \f$, the input to
   * ``beforeLayer`` becomes \f$ (x + f(x)) * \alpha \f$.
   *
   * @param[in] beforeLayer the index of the layer to which to scale the input
   * and residual connection output.
   * @param[in] scale the value by which to scale the sum of the previous layer
   * and output of the residual connection.
   */
  void addScale(int beforeLayer, float scale);

  /**
   * Adds a shortcut between two layers.
   *
   * @param[in] fromLayer the layer index from which the skip connection will
   * originate; must be in the range \f$ [0, N_{layers} - 1] \f$. If the index 0
   * is used, the input to the shortcut will be equal to the input to the
   * residual block.
   * @param[in] toLayer the layer index to which the skip connection outputs a
   * tensor; must be in the range \f$ [1, N_{layers} + 1] \f$. If the index
   * \f$ N_{layers} + 1 \f$ is used, the output of the shortcut will be added to
   * the output of the entire residual block.
   */
  void addShortcut(int fromLayer, int toLayer);

  /**
   * See ``Residual::addShortcut``.
   */
  template <typename T>
  void addShortcut(int fromLayer, int toLayer, const T& module) {
    addShortcut(fromLayer, toLayer, std::make_shared<T>(module));
  }

  /**
   * Adds a shortcut connection between two layers such that tensors passed
   * through the connection are forwarded through a passed module before being
   * added to the resultant module's input. Can be used to reshape the output of
   * input module to match the input dimensions for the output module.
   *
   * @param[in] fromLayer the layer index from which the shortcut connection
   * will originate; must be in the range \f$ [0, N_{layers} - 1] \f$. If the
   * index 0 is used, the input to the shortcut will be equal to the input to
   * the residual block.
   * @param[in] toLayer the layer index to which the shortcut connection outputs
   * a tensor; must be in the range \f$ [1, N_{layers} + 1] \f$. If the index
   * \f$ N_{layers} + 1 \f$  is used, the output of the shortcut will be added
   * to the output of the entire residual block.
   * @param[in] module a specified module through which the input to the
   * shortcut connection will be forwarded before being added to the input to
   * the destination module.
   */
  template <typename T>
  void addShortcut(int fromLayer, int toLayer, std::shared_ptr<T> module) {
    checkShortcut(fromLayer, toLayer);
    Container::add(module);
    processShortcut(fromLayer, toLayer, modules_.size() - 1);
    projectionsIndices_.insert(modules_.size() - 1);
  }

  std::vector<Variable> forward(const std::vector<Variable>& inputs) override;

  Variable forward(const Variable& input);

  std::string prettyString() const override;

  FL_BASIC_CONTAINER_CLONING(Residual)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Residual)
