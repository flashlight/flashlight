/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include "Module.h"

namespace fl {

/**
 * Applies the [sigmoid
 * function](https://en.wikipedia.org/wiki/Sigmoid_function) element-wise to a
 * `Variable`: \f[\text{sigmoid}(x) = \frac{1}{1 + e^{-x}}\f]
 */
class Sigmoid : public Module {
 public:
  Sigmoid();

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Module)
};

/**
 * Applies the [natural logarithm](https://en.wikipedia.org/wiki/Logarithm)
 * element-wise to a `Variable`.
 */
class Log : public Module {
 public:
  Log();

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Module)
};

/**
 * Applies the [hyperbolic tangent
 * function](https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent)
 * element-wise to a `Variable`:
 *\f[\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x +  e^{-x}}\f]
 */
class Tanh : public Module {
 public:
  Tanh();

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Module)
};

/**
 * Applies the hard-tanh function element-wise to a `Variable`:
 * \f[\text{HardTanh}(x) =
    \begin{cases}
      1 & \text{ if } x > 1 \\
      -1 & \text{ if } x < -1 \\
      x & \text{ otherwise } \\
    \end{cases}
    \f]
 */
class HardTanh : public Module {
 public:
  HardTanh();

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Module)
};

/**
 * Applies the [rectified linear
 * unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) function
 * element-wise to a `Variable`:
 * \f[ ReLU(x) = \max(0, x) \f]
 */
class ReLU : public Module {
 public:
  ReLU();

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(Module)
};

/**
 * Applies the [leaky rectified linear
 * unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
 function from [Maas et
 al
 (2013)](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf),
 _Rectifier Nonlinearities Improve Neural Network Acoustic Models_. Applied
 function element-wise to a `Variable`:
 * \f[
   \text{LeakyRELU}(x) =
    \begin{cases}
      x, & \text{ if } x \geq 0 \\
      \text{slope} \times x, & \text{ otherwise }
    \end{cases}
    \f]
 * where \f$\text{slope}\f$ is a constant by which the input will
 * be multiplied if less than zero.
 */
class LeakyReLU : public Module {
 private:
  double m_slope;

  FL_SAVE_LOAD_WITH_BASE(Module, m_slope)

 public:
  /**
   * Creates a `LeakyReLU` with the specified slope
   *
   * @param slope a constant by which the input will be multiplied if less than
   * 0
   */
  LeakyReLU(double slope = 0.0);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};

/**
 * Applies the [pramaeterized rectified linear
 unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
 function from [He et al (2015)](https://arxiv.org/pdf/1502.01852.pdf), _Delving
 Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet
 Classification_. Applied element-wise to a `Variable`, given some input size:
 * \f[
   \text{PReLU}(x) =
   \begin{cases}
     x, & \text{ if } x \geq 0 \\
     \text{value} \times x, & \text{ otherwise }
   \end{cases}
   \f]
 * where \f$\text{value}\f$ is a learned parameter whose initialization can be
 * tuned.
 */
class PReLU : public Module {
 private:
  PReLU() = default; // Intentionally private

  FL_SAVE_LOAD_WITH_BASE(Module)

 public:
  /**
   * Creates a `PReLU` with the specified value and input size
   *
   * @param value a constant by which the input will be multiplied if less than
   * 0
   * @param size the number of learnable parameters. The size must be a multiple
   * of the first dimension of the input
   */
  explicit PReLU(int size, double value = 0.25);

  /**
   * Creates a `PReLU` with a custom tensor; if the input is less than zero, the
   * output is equal to the tensor product of the input and this tensor. The
   * initialization for the learned tensor can be smaller than the input; it
   * will be broadcast in order to compute the product.
   *
   * @param w the tensor initializing the learned \f$\text{value}\f$ parameter
   */
  explicit PReLU(const Variable& w);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};

/**
 * Applies the [exponential linear
 unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) function
 from [Clevert et al (2015)](https://arxiv.org/pdf/1511.07289v1.pdf): _Fast and
 Accurate Deep Network Learning by Exponential Linear Units (ELUs)_. Applied
 element-wise to a `Variable`:
 *
 * \f[
   \text{ELU}(x) =
   \begin{cases}
     x & \text{ if } x \geq 0 \\
     \alpha \times (e^x - 1) & \text{ otherwise }
   \end{cases}
   \f]
 * where \f$\alpha\f$ is a tunable parameter.
 */
class ELU : public Module {
 private:
  double m_alpha;

  FL_SAVE_LOAD_WITH_BASE(Module, m_alpha)

 public:
  ELU(double alpha = 1.0);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};

/**
 * Applies the threshold rectified linear unit from [Konda et al
 * (2015)](https://arxiv.org/pdf/1402.3337.pdf): _Zero-bias autoencoders and the
 * benefits of co-adapting features_. Applied element-wise to a `Variable`:
 * \f[
   \text{ThresholdReLU}(x) =
   \begin{cases}
     x & \text{ if } x > \text{threshold} \\
     0 & \text{ otherwise }
   \end{cases}
   \f]
 * where \f$\text{threshold}\f$ is a tunable parameter.
 */
class ThresholdReLU : public Module {
 private:
  double m_threshold;

  FL_SAVE_LOAD_WITH_BASE(Module, m_threshold)

 public:
  /**
   * Creates a `ThresholdReLU` with the specified threshold.
   *
   * @param threshold the threshold value above which the unit returns the input
   */
  ThresholdReLU(double threshold = 1.0);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};

/**
 * Creates a Gated Linear Unit from
 [Dauphin et al  (2017)](https://arxiv.org/pdf/1612.08083.pdf): _Language
 Modeling with Gated Convolutional Networks_.
 * \f[\text{GLU}(x) = x_i \otimes \sigma(x_j)\f]
 * where \f$\otimes\f$ denotes the element-wise product \f$x_i\f$ is the
 * first half of the input, \f$x_j\f$ is the second half, and
 * \f$\sigma(x)\f$ is the sigmoid function.
 */
class GatedLinearUnit : public Module {
 private:
  int dim_;

  FL_SAVE_LOAD_WITH_BASE(Module, dim_)

 public:
  /**
   * Creates a `GatedLinearUnit`.
   *
   * @param dim the dimension along which the GLU will cut the input in half.
   * This dimension must be even in size in the input tensor.
   */
  GatedLinearUnit(int dim = 0);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};

/**
 * Applies the log softmax function to a tensor:
 * \f[
   \text{LogSoftmax}(x_i) =
   \log{\left (\frac{e^{x_i} }{ \sum_j e^{x_j}} \right)}
   \f]
 */
class LogSoftmax : public Module {
 private:
  int dim_;

  FL_SAVE_LOAD_WITH_BASE(Module, dim_)

 public:
  /**
   * Creates a `LogSoftmax`.
   *
   * @param dim the dimension along which to apply the LogSoftmax.
   */
  LogSoftmax(int dim = 0);

  Variable forward(const Variable& input) override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Sigmoid)
CEREAL_REGISTER_TYPE(fl::Tanh)
CEREAL_REGISTER_TYPE(fl::ReLU)
CEREAL_REGISTER_TYPE(fl::LeakyReLU)
CEREAL_REGISTER_TYPE(fl::PReLU)
CEREAL_REGISTER_TYPE(fl::ELU)
CEREAL_REGISTER_TYPE(fl::ThresholdReLU)
CEREAL_REGISTER_TYPE(fl::GatedLinearUnit)
CEREAL_REGISTER_TYPE(fl::Log)
CEREAL_REGISTER_TYPE(fl::HardTanh)
CEREAL_REGISTER_TYPE(fl::LogSoftmax)
