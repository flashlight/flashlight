/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/** A recurrent neural network (RNN) layer. The RNN layer supports several cell
 * types. The most basic RNN (e.g. an Elman network) computes the following
 * function:
 *  \f[ h_t = \sigma(W x_t + U h_{t-1} + b) \f]
 * If the RNN mode is RELU then \f$\sigma\f$ will be a `ReLU`. If the
 * RNN mode is TANH then it will be a `Tanh` function.
 *
 * Gated Recurrent Units (GRU) are supported. For details see the original [GRU
 * paper](https://arxiv.org/abs/1406.1078) or the [Wikipedia
 * page](https://en.wikipedia.org/wiki/Gated_recurrent_unit).
 *
 * LSTM cells are also supported (LSTM). The LSTM cell uses a forget
 * gate and does not have peephole connections. For details see the original
 * paper [Long Short-Term Memory](https://dl.acm.org/citation.cfm?id=1246450)
 * or the [Wikipedia
 * page](https://en.wikipedia.org/wiki/Long_short-term_memory).
 *
 * The input to the RNN is expected to be of shape [\f$X_{in}\f$, \f$N\f$,
 * \f$T\f$] where \f$N\f$ is the batch size and \f$T\f$ is the sequence length.
 *
 * The output of the RNN is will be of shape [\f$X_{out}\f$, \f$N\f$, \f$T\f$].
 * Here \f$X_{out}\f$ will be `hidden_size` if the RNN is unidirectional and it
 * will be twice the `hidden_size` if the RNN is bidirectional.
 *
 * In addition the RNN supports including the hidden state and the cell state
 * as input and output. When these are input as the empty Variable they are
 * assumed to be zero.
 */
class RNN : public Module {
 private:
  RNN() = default; // Intentionally private

  int inputSize_;
  int hiddenSize_;
  int numLayers_;
  RnnMode mode_;
  bool bidirectional_;
  float dropProb_;

  FL_SAVE_LOAD_WITH_BASE(
      Module,
      inputSize_,
      hiddenSize_,
      numLayers_,
      mode_,
      bidirectional_,
      dropProb_)

  void initialize();

 public:
  /** Construct an RNN layer.
   * @param input_size The dimension of the input (e.g. \f$X_{in}\f$)
   * @param hidden_size The hidden dimension of the RNN.
   * @param num_layers The number of recurrent layers.
   * @param mode The RNN mode to use. Can be any of:
   * - RELU
   * - TANH
   * - LSTM
   * - GRU
   * @param bidirectional Whether or not the RNN is bidirectional. If `true` the
   * output dimension will be doubled.
   * @param drop_prob The probability of dropout after each RNN layer except the
   * last layer.
   */
  RNN(int input_size,
      int hidden_size,
      int num_layers,
      RnnMode mode,
      bool bidirectional = false,
      float drop_prob = 0.0);

  std::vector<Variable> forward(const std::vector<Variable>& inputs) override;

  using Module::operator();

  /** Forward the RNN Layer.
   * @param input Should be of shape [\f$X_{in}\f$, \f$N\f$, \f$T\f$]
   * @returns a single output Variable with shape [\f$X_{out}\f$, \f$N\f$,
   * \f$T\f$]
   */
  Variable forward(const Variable& input);

  Variable operator()(const Variable& input);

  /** Forward the RNN Layer.
   * @param input Should be of shape [\f$X_{in}\f$, \f$N\f$, \f$T\f$]
   * @param hidden_state Should be of shape [\f$X_{out}\f$, \f$N\f$]. If an
   * empty Variable is passed in then the hidden state is assumed zero.
   * @returns An tuple of output Variables.
   *  - The first element is the output of the RNN of shape [\f$X_{out}\f$,
   *  \f$N\f$, \f$T\f$]
   *  - The second element is the hidden state of the RNN of shape
   *  [\f$X_{out}\f$, \f$N\f$]
   */
  std::tuple<Variable, Variable> forward(
      const Variable& input,
      const Variable& hidden_state);

  std::tuple<Variable, Variable> operator()(
      const Variable& input,
      const Variable& hidden_state);

  /** Forward the RNN Layer.
   * @param input Should be of shape [\f$X_{in}\f$, \f$N\f$, \f$T\f$]
   * @param hidden_state Should be of shape [\f$X_{out}\f$, \f$N\f$]. If an
   * empty Variable is passed in then the hidden state is assumed zero.
   * @param cell_state Should be of shape [\f$X_{out}\f$, \f$N\f$]. If an empty
   * Variable is passed in then the hidden state is assumed zero.
   * @returns An tuple of output Variables.
   *  - The first element is the output of the RNN of shape [\f$X_{out}\f$,
   *  \f$N\f$, \f$T\f$]
   *  - The second element is the hidden state of the RNN of shape
   *  [\f$X_{out}\f$, \f$N\f$]
   *  - The third element is the cell state of the RNN of shape [\f$X_{out}\f$,
   *  \f$N\f$]
   */
  std::tuple<Variable, Variable, Variable> forward(
      const Variable& input,
      const Variable& hidden_state,
      const Variable& cell_state);

  std::tuple<Variable, Variable, Variable> operator()(
      const Variable& input,
      const Variable& hidden_state,
      const Variable& cell_state);

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::RNN)
