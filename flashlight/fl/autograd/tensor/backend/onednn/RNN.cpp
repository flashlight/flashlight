/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/onednn/OneDnnAutogradExtension.h"

#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <dnnl.hpp>

#include "flashlight/fl/autograd/tensor/backend/onednn/DnnlUtils.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {
namespace {

struct ParsedWeightsAndBias {
  // First layer - will be empty if inSize == hiddenSize
  Tensor weightsInput1L;
  Tensor weightsHidden1L;
  Tensor bias1L;
  // All other layers
  Tensor weightsInput;
  Tensor weightsHidden;
  Tensor bias;
};

// Each gate's weights have dimensions d1 x d2
Tensor reorderLbrGruWeights(int d1, int d2, const Tensor& weights) {
  // LBR GRU requires switch the given the r, u, o gate order from cuDNN to u,
  // r, o as required by oneDNN (this from empirical verification)
  int weightsSize = d1 * d2;
  if (weights.elements() != weightsSize * 3) {
    throw std::invalid_argument(
        "RNN reorderLbrGruWeights given invalid weights tensor or dims - "
        "weights of size " +
        std::to_string(weights.elements()) + " which should be exactly " +
        std::to_string(weightsSize * 3));
  }
  return fl::concatenate(
      0,
      weights.flat(fl::range(weightsSize, 2 * weightsSize)),
      weights.flat(fl::range(0, weightsSize)),
      weights.flat(fl::range(2 * weightsSize, fl::end)));
}

/**
 * Converts flat cuDNN weights into the corresponding oneDNN onednn RNN weights.
 */
ParsedWeightsAndBias parseWeights(
    const Tensor& weights,
    RnnMode mode,
    int numLayers,
    int directionMult,
    int inSize,
    int numGates,
    int hiddenSize) {
  ParsedWeightsAndBias out;

  // Per-layer sizes for weightsInput and weightsHidden.
  // If inSize == hiddenSize, then weightsInputSize == weightsHiddenSize for all
  // layers, else all but the first layer
  int weightsInputSize1L = directionMult * inSize * numGates * hiddenSize;
  int weightsHiddenSize = directionMult * hiddenSize * numGates * hiddenSize;
  int weightsInputSize = weightsHiddenSize;
  int lbrGruBias = mode == RnnMode::GRU ? 1 : 0;
  int biasSize =
      numLayers * directionMult * (numGates + lbrGruBias) * hiddenSize;

  bool firstLayerDifferent = inSize != hiddenSize;
  // Adjusted if skipping first layer parsing
  int numWeightsLayers = firstLayerDifferent ? numLayers - 1 : numLayers;
  int weightsOffset =
      firstLayerDifferent ? weightsInputSize1L + weightsHiddenSize : 0;
  // If skipping the first layer, parse then skip over the first layer
  // weights and parse the remaining layers. Parsing all bias layers is still
  // fine since biases for each layer have the same size
  if (firstLayerDifferent) {
    out.weightsInput1L = weights.flat(fl::range(weightsInputSize1L));
    out.weightsHidden1L = weights.flat(
        fl::range(weightsInputSize1L, weightsInputSize1L + weightsHiddenSize));

    if (mode == RnnMode::GRU) {
      out.weightsInput1L =
          reorderLbrGruWeights(inSize, hiddenSize, out.weightsInput1L);
      out.weightsHidden1L =
          reorderLbrGruWeights(hiddenSize, hiddenSize, out.weightsHidden1L);
    }
  }

  auto weightsFlat = weights.flatten().astype(weights.type());
  // cuDNN RNN weights, for each layer, are arranged with a chunk of
  // input-hidden weights for each layer followed by a chunk of hidden-hidden
  // weights for each layer:
  // {[layers x [hiddenSize, inputSize]], [layers x  [hiddenSize, hiddenSize]] }
  // Rearrange this to what oneDNN expects (or will reorder if not optimal),
  // which is numLayers chunks of two chunks containing input-hidden and
  // hidden-hidden:
  // {[layers x [[hiddenSize x inSize], [hiddenSize x hiddenSize]]]}
  // Note that the loop is over the total number of layers in case we'r doing a
  // single-layer operation where input size and hidden size are different but
  // we'll call another primitive with the output of that first layer as the
  // input to the next layers
  auto weightsInput = Tensor({0}, weights.type());
  auto weightsHidden = Tensor({0}, weights.type());
  Tensor weightsFlatOffset =
      weightsFlat.flat(fl::range(weightsOffset, fl::end));
  // Specifically ignore the first layer's weights, so inSize == hiddenSize
  for (int i = 0; i < numWeightsLayers; ++i) {
    // number of input/hidden weights
    // TODO: Will change for bidirectional
    int chunkSize = hiddenSize * hiddenSize * numGates;
    // weights per layer
    int layerChunkSize = chunkSize + chunkSize;

    // Grab input-hidden weights and chunk them together
    auto inputWeightsChunk = weightsFlatOffset.flat(
        fl::range(layerChunkSize * i, layerChunkSize * i + chunkSize));
    // Grab hidden-hidden weights and chunk them together
    auto inputHiddenChunk = weightsFlatOffset.flat(fl::range(
        layerChunkSize * i + chunkSize,
        layerChunkSize * i + chunkSize + chunkSize));

    if (mode == RnnMode::GRU) {
      inputWeightsChunk =
          reorderLbrGruWeights(hiddenSize, hiddenSize, inputWeightsChunk);
      inputHiddenChunk =
          reorderLbrGruWeights(hiddenSize, hiddenSize, inputHiddenChunk);
    }

    weightsInput = fl::concatenate(2, weightsInput, inputWeightsChunk);
    weightsHidden = fl::concatenate(2, weightsHidden, inputHiddenChunk);
  }
  out.weightsInput = weightsInput;
  out.weightsHidden = weightsHidden;

  // Reduce the weights to form biases. cuDNN uses two separate bias terms:
  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNMode_t -
  // oneDNN expects only one bias term. Sum together the coefficients for both
  // bias terms to get a single bias term for oneDNN. The gradients for
  // each term can be computed as one since the gradients with respect to
  // the bias subarrays will simply be half of the computed gradient with
  // oneDNN
  Tensor bias(weights.type());
  int biasStartOffset = numLayers * weightsHiddenSize +
      (numLayers - 1) * weightsInputSize + weightsInputSize1L;
  // In vanilla RNN modes, the biases can be simply added:
  // two biases for each bias in fl cuDNN with CUDNN_RNN_DOUBLE_BIAS (default)
  int numBiases = 2;
  // First, grab a subarray which contains only both bias terms; then add them
  Tensor biasFlat = weightsFlat.flat(fl::range(biasStartOffset, fl::end));
  // Layout is: {numLayers x [numBiases x [bias shape]]}
  for (int i = 0; i < numLayers; ++i) {
    if (mode == RnnMode::GRU) {
      int lbrGruChunkSize = hiddenSize * 6;
      // In the case of the LBR GRU, there's an extra bias term which shouldn't
      // be combined with the first two pairs of biases. Six chunks total.
      // cuDNN --> oneDNN transformation for ordering:
      // r1, u1, o, r2, u2, u' --> u1 + u2, r1 + r2, o, u'
      int base = i * lbrGruChunkSize;
      // The sum of the following tensors yields the correct bias
      // u1, r1, o, u'
      auto biases1 = fl::concatenate(
          0,
          // u1 -- [1, 2]
          biasFlat.flat(
              fl::range(base + hiddenSize * 1, base + hiddenSize * 2)),
          // r1 -- [0, 1]
          biasFlat.flat(
              fl::range(base + hiddenSize * 0, base + hiddenSize * 1)),
          // o -- [2, 3]
          biasFlat.flat(
              fl::range(base + hiddenSize * 2, base + hiddenSize * 3)),
          // 'u -- [5, 6]
          biasFlat.flat(
              fl::range(base + hiddenSize * 5, base + hiddenSize * 6)));
      // u2, r2, 0, 0
      auto biases2 = fl::concatenate(
          0,
          // u2 -- [4, 5]
          biasFlat.flat(
              fl::range(base + hiddenSize * 4, base + hiddenSize * 5)),
          // r2 -- [3, 4]
          biasFlat.flat(
              fl::range(base + hiddenSize * 3, base + hiddenSize * 4)),
          // zeroes to add to o and u'
          fl::full({hiddenSize * 2}, 0., biasFlat.type()));
      auto layerBiasCombined = biases1 + biases2;
      bias = fl::concatenate(0, bias, layerBiasCombined);
    } else {
      // The number of bias terms in the tensor per-layer
      int layerStride = biasSize / numLayers * numBiases;
      auto biases1 = biasFlat(fl::range(
          layerStride * i, layerStride * i + layerStride / numBiases));
      auto biases2 = biasFlat(fl::range(
          layerStride * i + layerStride / numBiases, layerStride * (i + 1)));
      auto layerBiasCombined = biases1 + biases2;
      bias = fl::concatenate(0, bias, layerBiasCombined);
    }
  }

  if (firstLayerDifferent) {
    out.bias1L = bias.flat(fl::range(biasSize / numLayers));
    if (numLayers > 1) {
      // bias for the second --> last layer
      bias = bias.flat(fl::range(biasSize / numLayers, fl::end));
    }
  }
  out.bias = bias;

  // Case for a single layer of different in/hidden size
  if (firstLayerDifferent && numLayers == 1) {
    out.weightsInput = out.weightsInput1L;
    out.weightsHidden = out.weightsHidden1L;
    out.bias = out.bias1L;
  }

  return out;
}

struct RnnResult {
  dnnl::memory workspace;
  Tensor y; // output
  Tensor hy; // hidden output
  Tensor cy; // cell output
};

/*
 * Does forward for a single onednn RNN primitive
 */
RnnResult rnnImpl(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weightsInput,
    const Tensor& weightsHidden,
    const Tensor& bias,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    dnnl::algorithm activation,
    int numGates,
    dnnl::rnn_direction direction,
    int directionMult,
    dnnl::prop_kind kind,
    float dropout) {
  RnnResult result;
  auto dnnlEngine = detail::DnnlEngine::getInstance().getEngine();

  // Dimensions
  int inSize = input.dim(0);
  int batchSize = input.ndim() < 2 ? 1 : input.dim(1);
  int seqLength = input.ndim() < 3 ? 1 : input.dim(2);
  dnnl::memory::dims inputDims = {seqLength, batchSize, inSize};
  dnnl::memory::dims outputDims = {
      seqLength, batchSize, hiddenSize * directionMult};
  auto dType = detail::dnnlMapToType(input.type());
  int totalLayers = numLayers;
  int outSize = hiddenSize;
  dnnl::memory::dims hDims = {
      totalLayers, directionMult, batchSize, hiddenSize};
  dnnl::memory::dims cDims = {
      totalLayers, directionMult, batchSize, hiddenSize};
  int extraBias = mode == RnnMode::GRU ? 1 : 0; // for LBR GRU
  dnnl::memory::dims biasDims = {
      numLayers, directionMult, numGates + extraBias, hiddenSize};
  // ldigo
  dnnl::memory::dims weightsInputDims = {
      numLayers, directionMult, inSize, numGates, hiddenSize};
  dnnl::memory::dims weightsHiddenDims = {
      numLayers, directionMult, hiddenSize, numGates, hiddenSize};

  // Out tensors: output (y), hidden state output (hy), cell state output (cy)
  auto y = Tensor({outSize, batchSize, seqLength}, input.type());
  auto hy = Tensor({hiddenSize, batchSize, totalLayers}, input.type());
  Tensor cy;
  if (mode == RnnMode::LSTM) {
    cy = Tensor(hy.shape(), input.type());
  }

  // Memory for forward
  auto tnc = dnnl::memory::format_tag::tnc;
  auto ldnc = dnnl::memory::format_tag::ldnc;
  auto ldgoi = dnnl::memory::format_tag::ldgoi;
  auto ldgo = dnnl::memory::format_tag::ldgo;
  const detail::DnnlMemoryWrapper inputMemInit(
      input.asContiguousTensor(), {inputDims}, tnc);
  const detail::DnnlMemoryWrapper outputMemInit(y, {outputDims}, tnc);
  detail::DnnlMemoryWrapper hiddenInMemInit;
  if (!hiddenState.isEmpty()) {
    hiddenInMemInit = detail::DnnlMemoryWrapper(
        hiddenState.asContiguousTensor(), {hDims}, ldnc);
  }
  const detail::DnnlMemoryWrapper hiddenOutMemInit(hy, {hDims}, ldnc);
  const detail::DnnlMemoryWrapper weightsInputMemRawInit(
      weightsInput.asContiguousTensor(), {weightsInputDims}, ldgoi);
  const detail::DnnlMemoryWrapper weightsHiddenMemRawInit(
      weightsHidden.asContiguousTensor(), {weightsHiddenDims}, ldgoi);
  const detail::DnnlMemoryWrapper biasMemInit(
      bias.asContiguousTensor(), {biasDims}, ldgo);

  // TODO(jacobkahn): don't force a format tag - use any and do a reorder based
  // on the format of the primitive - what it says - like you're supposed to
  // Primitive for reordering input weights: ldgoi --> ldigo
  auto weightsInputMemDesc = dnnl::memory::desc(
      weightsInputDims, dType, dnnl::memory::format_tag::ldigo);
  auto weightsInputMemInit = dnnl::memory(weightsInputMemDesc, dnnlEngine);
  // Primitive for reordering iter/hidden weights: ldgoi --> ldigo
  auto weightsHiddenMemDesc = dnnl::memory::desc(
      weightsHiddenDims, dType, dnnl::memory::format_tag::ldigo);
  auto weightsHiddenMemInit = dnnl::memory(weightsHiddenMemDesc, dnnlEngine);

  // Add arguments
  std::unordered_map<int, dnnl::memory> rnnFwdArgs = {
      {DNNL_ARG_SRC_LAYER, inputMemInit.getMemory()},
      {DNNL_ARG_SRC_ITER, hiddenInMemInit.getMemory()},
      {DNNL_ARG_WEIGHTS_LAYER, weightsInputMemInit},
      {DNNL_ARG_WEIGHTS_ITER, weightsHiddenMemInit},
      {DNNL_ARG_BIAS, biasMemInit.getMemory()},
      {DNNL_ARG_DST_LAYER, outputMemInit.getMemory()},
      {DNNL_ARG_DST_ITER, hiddenOutMemInit.getMemory()}};

  // Workspace memory, if needed
  dnnl::memory workspace;
  std::vector<dnnl::primitive> network;
  std::vector<std::unordered_map<int, dnnl::memory>> fwdArgs;

  // reorder input weights
  network.push_back(
      dnnl::reorder(weightsInputMemRawInit.getMemory(), weightsInputMemInit));
  fwdArgs.push_back(
      {{DNNL_ARG_FROM, weightsInputMemRawInit.getMemory()},
       {DNNL_ARG_TO, weightsInputMemInit}});
  // reorder iter weights
  network.push_back(
      dnnl::reorder(weightsHiddenMemRawInit.getMemory(), weightsHiddenMemInit));
  fwdArgs.push_back(
      {{DNNL_ARG_FROM, weightsHiddenMemRawInit.getMemory()},
       {DNNL_ARG_TO, weightsHiddenMemInit}});

  // Initialize descriptors
  if (mode == RnnMode::RELU || mode == RnnMode::TANH) {
    auto vanillaPd = dnnl::vanilla_rnn_forward::primitive_desc(
        dnnlEngine,
        kind,
        activation,
        direction,
        inputMemInit.getDescriptor(),
        hiddenInMemInit.getDescriptor(),
        weightsInputMemDesc, // weights "layer"
        weightsHiddenMemDesc, // weights "iter"
        biasMemInit.getDescriptor(),
        outputMemInit.getDescriptor(),
        hiddenOutMemInit.getDescriptor());
    network.push_back(dnnl::vanilla_rnn_forward(vanillaPd));
    workspace = dnnl::memory(vanillaPd.workspace_desc(), dnnlEngine);

  } else if (mode == RnnMode::LSTM) {
    // LSTM-only
    // input cell state
    // TODO(jacobkahn): function that takes the array and
    // returns the desciptor and memory -- takes an argument for
    // which determines whether or not it's ok to return empty
    // descriptors if the array is empty
    detail::DnnlMemoryWrapper cellInMemInit;
    if (!cellState.isEmpty()) {
      cellInMemInit = detail::DnnlMemoryWrapper(
          cellState.asContiguousTensor(), {cDims}, ldnc);
    }
    // output cell state
    detail::DnnlMemoryWrapper cellOutMemInit(cy, cDims, ldnc);

    auto lstmPd = dnnl::lstm_forward::primitive_desc(
        dnnlEngine,
        kind,
        direction,
        inputMemInit.getDescriptor(),
        hiddenInMemInit.getDescriptor(),
        cellInMemInit.getDescriptor(),
        weightsInputMemDesc, // weights "layer"
        weightsHiddenMemDesc, // weights "iter"
        biasMemInit.getDescriptor(),
        outputMemInit.getDescriptor(),
        hiddenOutMemInit.getDescriptor(),
        cellOutMemInit.getDescriptor());
    network.push_back(dnnl::lstm_forward(lstmPd));
    workspace = dnnl::memory(lstmPd.workspace_desc(), dnnlEngine);
    rnnFwdArgs.insert({DNNL_ARG_SRC_ITER_C, cellInMemInit.getMemory()});
    rnnFwdArgs.insert({DNNL_ARG_DST_ITER_C, cellOutMemInit.getMemory()});

  } else if (mode == RnnMode::GRU) {
    // Use a linear-before-reset GRU so we can have parity with cuDNN
    auto gruPd = dnnl::lbr_gru_forward::primitive_desc(
        dnnlEngine,
        kind,
        direction,
        inputMemInit.getDescriptor(),
        hiddenInMemInit.getDescriptor(),
        weightsInputMemDesc,
        weightsHiddenMemDesc,
        biasMemInit.getDescriptor(),
        outputMemInit.getDescriptor(),
        hiddenOutMemInit.getDescriptor());
    network.push_back(dnnl::lbr_gru_forward(gruPd));
    workspace = dnnl::memory(gruPd.workspace_desc(), dnnlEngine);
  }
  rnnFwdArgs.insert({DNNL_ARG_WORKSPACE, workspace});
  fwdArgs.push_back(rnnFwdArgs);

  detail::executeNetwork(network, fwdArgs);

  result.y = y;
  result.hy = hy;
  result.cy = cy;
  result.workspace = workspace;
  return result;
}

} // namespace

std::tuple<Tensor, Tensor, Tensor> OneDnnAutogradExtension::rnn(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const int hiddenSize,
    const int numLayers,
    const RnnMode mode,
    const bool bidirectional,
    const float dropout,
    std::shared_ptr<detail::AutogradPayload> autogradPayload) {
  if (dropout > 0.0) {
    throw std::invalid_argument("onednn RNN: dropout > 0.0 unsupported");
  }
  if (bidirectional) {
    throw std::invalid_argument("onednn RNN: bidirectional not yet supported");
  }

  const bool train = (autogradPayload != nullptr);

  // Constants
  auto direction = bidirectional
      ? dnnl::rnn_direction::bidirectional_concat
      : dnnl::rnn_direction::unidirectional_left2right;
  int directionMult = bidirectional ? 2 : 1;
  auto kind = train ? dnnl::prop_kind::forward_training
                    : dnnl::prop_kind::forward_inference;
  int numGates = 1;
  auto activation = dnnl::algorithm::undef;
  switch (mode) {
    case RnnMode::LSTM:
      numGates = 4;
      break;
    case RnnMode::GRU:
      numGates = 3;
      break;
    case RnnMode::RELU:
      activation = dnnl::algorithm::eltwise_relu;
      break;
    case RnnMode::TANH:
      activation = dnnl::algorithm::eltwise_tanh;
      break;
    default:
      break;
  }

  int inSize = input.dim(0);

  // In Flashlight, all RNN weights are stored as one contiguous tensor, so we
  // have to parse out the input weights, input biases, hidden weights, and
  // hidden biases from one tensor. Order doesn't matter since the arrangement
  // is a black box
  auto parsedWeights = parseWeights(
      weights, mode, numLayers, directionMult, inSize, numGates, hiddenSize);

  RnnResult result;
  // The oneDNN RNN primitive has an API limitation where input size and
  // hidden size can only differ if the primitive has exactly one layer.
  // Therefore, for computations for more than one layer, first do the
  // operation for one layer, which gives an output vector of size [hidden
  // size, batch size, sequence length * number of directions], then use
  // that output as the input for layers [2, L]. Since the input size dim 0
  // is now the hidden size, the primitive can fuse computation for
  // arbitrarily-many layers.
  if (input.dim(0) == hiddenSize || numLayers == 1) {
    // Input and hidden size are the same, or we only have one layer, which
    // means we can call the impl as is and parse weights "normally"
    result = rnnImpl(
        input,
        hiddenState,
        cellState,
        parsedWeights.weightsInput,
        parsedWeights.weightsHidden,
        parsedWeights.bias,
        hiddenSize,
        numLayers,
        mode,
        activation,
        numGates,
        direction,
        directionMult,
        kind,
        dropout);
  } else {
    // We require more than one layer with different input and hidden states -
    // see the above. Seek to the first layer's hidden/cell state, weights, and
    // bias
    RnnResult resultL1 = rnnImpl(
        input,
        hiddenState(fl::span, fl::span, 0),
        cellState(fl::span, fl::span, 0),
        parsedWeights.weightsInput1L,
        parsedWeights.weightsHidden1L,
        parsedWeights.bias1L,
        hiddenSize,
        1,
        mode,
        activation,
        numGates,
        direction,
        directionMult,
        kind,
        dropout);

    /* Layers [2..N] */
    // Seek  past the first layer's hidden/cell state, weights, and bias
    RnnResult resultL2N = rnnImpl(
        resultL1.y, // fixme
        hiddenState(fl::span, fl::span, fl::range(1, fl::end)),
        cellState(fl::span, fl::span, fl::range(1, fl::end)),
        parsedWeights.weightsInput,
        parsedWeights.weightsHidden,
        parsedWeights.bias,
        hiddenSize,
        numLayers - 1, // layers [2..N]
        mode,
        activation,
        numGates,
        direction,
        directionMult,
        kind,
        dropout);

    result.y = resultL2N.y;
    result.hy = fl::concatenate(2, resultL1.hy, resultL2N.hy);
    result.cy = fl::concatenate(2, resultL1.cy, resultL2N.cy);
  }

  return {result.y, result.hy, result.cy};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> OneDnnAutogradExtension::rnnBackward(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const std::shared_ptr<detail::RNNGradData> gradData,
    const Tensor& output,
    const int numLayers,
    const int hiddenSize,
    const RnnMode mode,
    const bool bidirectional,
    const float dropProb,
    const std::shared_ptr<detail::AutogradPayload> payload) {
  throw std::runtime_error(
      "onednn RNN: Gradient computation not yet supported");
}

} // namespace fl
