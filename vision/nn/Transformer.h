#pragma once
#include "flashlight/nn/nn.h"
#include "vision/dataset/BoxUtils.h"
#include "iostream"

#include <cassert>


namespace fl {
namespace cv {


fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std);
}

fl::Variable transformerRotate(const fl::Variable& input) {
  auto data = input.array();
  int d0 = data.dims(0);
  int d1 = data.dims(1);
  int d2 = data.dims(2);
  int d3 = data.dims(3);
  data = af::join(0, data, af::constant(0.0, d1, d1, d2, d3));
  data = af::moddims(data, af::dim4((d0 + d1) * d1, 1, d2, d3));
  data = data.rows(0, (d1 + d0 - 1) * d1 - 1);
  data = af::moddims(data, af::dim4(d0 + d1 - 1, d1, d2, d3));
  auto gradFunc = [d0, d1, d2, d3](
                      std::vector<fl::Variable>& inputs,
                      const fl::Variable& gradOutput) {
    auto gradData = gradOutput.array();
    gradData = af::moddims(gradData, af::dim4((d0 + d1 - 1) * d1, 1, d2, d3));
    gradData = af::join(0, gradData, af::constant(0.0, d1, 1, d2, d3));
    gradData = af::moddims(gradData, af::dim4(d0 + d1, d1, d2, d3));
    gradData = gradData.rows(0, d0 - 1);
    inputs[0].addGrad(fl::Variable(gradData, false));
  };
  return fl::Variable(data, {input}, gradFunc);
}

fl::Variable transformerMultiheadAttention(
    const fl::Variable& query,
    const fl::Variable& key,
    const fl::Variable& value,
    const fl::Variable& posEmb,
    const fl::Variable& mask,
    const int32_t nHead,
    const double pDropout,
    const int32_t offset = 0) {
  int32_t bsz = query.dims(2);
  int32_t modelDim = query.dims(1);
  int32_t headDim = modelDim / nHead;

  auto q = moddims(query, af::dim4(-1, headDim, nHead * bsz));
  auto k = moddims(key, af::dim4(-1, headDim, nHead * bsz));
  auto v = moddims(value, af::dim4(-1, headDim, nHead * bsz));

  auto scores = matmulNT(q, k);
  if (!posEmb.isempty()) {
    int n = posEmb.dims(0) / 2 - offset;
    auto pscores = transformerRotate(matmulNT(posEmb, q));
    scores = scores + transpose(pscores.rows(n, n + k.dims(0) - 1));
  }
  scores = scores / std::sqrt(float(headDim));
  if (!mask.isempty()) {
    scores = scores + tileAs(mask, scores);
  }

  auto attn = dropout(softmax(scores, 1), pDropout);
  auto result = matmul(attn, v);
  result = moddims(result, af::dim4(-1, headDim * nHead, bsz));
  return result;
}

class MultiheadAttention : public Container {
  public:
    MultiheadAttention(
        int32_t modelDim,
        int32_t headDim,
        int32_t numHeads,
         float pDropout=0.f
    ) : pDropout_(pDropout),
      numHeads_(numHeads){
      wq_ = std::make_shared<Linear>(
        transformerInitLinear(modelDim, headDim * numHeads));
      wk_ = std::make_shared<Linear>(
        transformerInitLinear(modelDim, headDim * numHeads));
      wv_ = std::make_shared<Linear>(
        transformerInitLinear(modelDim, headDim * numHeads));
      wf_ = std::make_shared<Linear>(
          transformerInitLinear(headDim * numHeads, modelDim));
      add(wq_);
      add(wk_);
      add(wv_);
      add(wf_);
  };

  std::vector<Variable> forward(
      const Variable queries,
      const Variable keys,
      const Variable values) {
      int n = queries.dims(1), bsz = queries.dims(2);
      double pDrop = train_ ? pDropout_ : 0.0;

      auto q = transpose((*wq_)(queries));
      auto k = transpose((*wk_)(keys));
      auto v = transpose((*wv_)(values));

      //Variable mask, posEmb;
      //posEmb = tile(params_[0], af::dim4(1, 1, nHeads_ * bsz));
      //if (useMask_ && input.back().dims(1) > 1) {
        //mask = getMask(n, input.size() == 2);
      //}

      //int offset = (input.size() == 1) ? 0 : input[0].dims(1);
      auto posEmb = fl::Variable();
      auto mask = fl::Variable();
      int offset = 0;
      auto result = transformerMultiheadAttention(
        q, k, v, posEmb, mask, numHeads_, pDropout_);
      std::vector<Variable> results = { (*wf_)(transpose(result)) };
      return results;
  }

  std::vector<Variable> forward(const std::vector<Variable>& input) override {
    return this->forward(input[0], input[1], input[2]);
  }

  std::string prettyString() const override {
    return "MultiheadAttention";
  }


  protected:
  std::shared_ptr<Linear> wq_;
  std::shared_ptr<Linear> wk_;
  std::shared_ptr<Linear> wv_;
  std::shared_ptr<Linear> wf_;
  float pDropout_;
  int32_t numHeads_;
};

class TransformerBaseLayer : public Container {
 public:
  TransformerBaseLayer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout
      ) :
      self_attn_(std::make_shared<MultiheadAttention>(modelDim, modelDim, nHeads, pDropout)),
      w1_(std::make_shared<Linear>(transformerInitLinear(modelDim, mlpDim))),
      w2_(std::make_shared<Linear>(transformerInitLinear(mlpDim, modelDim))),
      norm1_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))),
      norm2_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))),
      pDropout_(pDropout)
      {
        add(self_attn_);
        add(w1_);
        add(w2_);
        add(norm1_);
        add(norm2_);
      };


  Variable mlp(const Variable& in) {
    float pDropout = train_ ? pDropout_ : 0.0;
    return (*w2_)(dropout(relu((*w1_)(in)), pDropout));
  }

  Variable withPosEmbed(const Variable& input, const Variable& pos) {
    if (pos.isempty()) {
      return input;
    }
    return input + pos;
  }

  Variable selfAttention(
      const Variable& input,
      const Variable &pos) {
    auto k = withPosEmbed(input, pos);
    auto q = withPosEmbed(input, pos);
    return self_attn_->forward(q, k, input)[0];
  }

 protected:
  std::shared_ptr<MultiheadAttention> self_attn_;
  std::shared_ptr<Linear> w1_, w2_;
  std::shared_ptr<LayerNorm> norm1_, norm2_;
  TransformerBaseLayer();
  float pDropout_;

};

class TransformerEncoderLayer : public TransformerBaseLayer {
 public:
  TransformerEncoderLayer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout) :
      TransformerBaseLayer(modelDim, headDim, mlpDim, nHeads, pDropout)
      { };

  std::vector<Variable> forward(const std::vector<Variable>& input) override {
    auto src = input[0];
    // Self Attention
    {
      auto src2 = (*norm1_)(src);
      src2 = this->selfAttention(src, fl::Variable());
      src = src + dropout(src2, pDropout_);
    }
    // MLP
    {
      auto src2 = norm2_->forward(src);
      src2 = mlp(src2);
      src = src + dropout(src2, pDropout_);
    }
    return { src };
  }

  std::string prettyString() const override {
    return "TransformerEncoderLayer";
  }

};

class TransformerDecoderLayer : public TransformerBaseLayer {
 public:
  TransformerDecoderLayer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout) :
        TransformerBaseLayer(modelDim, headDim, mlpDim, nHeads, pDropout),
        encoder_attn_(std::make_shared<MultiheadAttention>(modelDim, modelDim, nHeads, pDropout)),
        norm3_(std::make_shared<LayerNorm>(std::vector<int>({0, 3})))
        { };

  std::vector<Variable> forward(const std::vector<Variable>& input) override {
    auto tgt = input[0];
    auto memory = input[0];
    // Self attention
    {
      auto tgt2 = (*norm1_)(tgt);
      tgt2 = this->selfAttention(tgt2, fl::Variable());
      tgt = tgt + dropout(tgt2, pDropout_);
    }
    // Encoder-decoder attention
    {
      auto tgt2 = (*norm2_)(tgt);
      tgt2 = encoder_attn_->forward({ 
          tgt2, // queries
          memory, // keys
          memory // values
          })[0];
      // TODO fix norms
      tgt = tgt + dropout(tgt2, pDropout_);
    }
    // MLP
    {
      auto tgt2 = (*norm3_)(tgt);
      tgt2 = mlp(tgt2);
      tgt = tgt + tgt2;
    }
    return { tgt };
  }

  std::string prettyString() const override {
    return "TransformerDecoderLayer";
  }

private:
  std::shared_ptr<MultiheadAttention> encoder_attn_;
  std::shared_ptr<LayerNorm> norm3_;
};

class TransformerDecoder : public Container {
  public:
    TransformerDecoder(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      int32_t layers,
      float pDropout
      ) {
      // TODO add norm
      for(int i = 0; i < layers; i++) {
        add(TransformerDecoderLayer(modelDim, headDim, mlpDim, nHeads, pDropout));
      }
    }

    std::vector<Variable> forward(const std::vector<Variable>& input) override {
      auto memory = input[1];
      auto tgt = input[0];

      auto output = tgt;
      std::cout << "modules " << modules().size() << std::endl;
      for(auto mod : modules()) {
        af_print(output.array());
        output = mod->forward({output, memory})[0];
      }
      return { output };
    }
  std::string prettyString() const override {
    return "TransformerDecoder";
  }
};

class TransformerEncoder : public Container {
  public:
    TransformerEncoder(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      int32_t layers,
      float pDropout) {
      // TODO add norm
      for(int i = 0; i < layers; i++) {
        add(TransformerEncoderLayer(modelDim, headDim, mlpDim, nHeads, pDropout));
      }
    }

    std::vector<Variable> forward(const std::vector<Variable>& input) override {
      auto output = input;
      for(auto mod : modules_) {
        output = mod->forward(output);
      }
      return output;
    }

    std::string prettyString() const override {
      return "TransformerDecoder";
    }
};

class Transformer : public Container {
public:

  Transformer(
      int32_t modelDim,
      int32_t numHeads,
      int32_t numEncoderLayers,
      int32_t numDecoderLayers,
      int32_t mlpDim,
      float pDropout) :
    encoder_(std::make_shared<TransformerEncoder>(modelDim, modelDim / numHeads, mlpDim, numHeads, numEncoderLayers, pDropout)),
    decoder_(std::make_shared<TransformerDecoder>(modelDim, modelDim / numHeads, mlpDim, numHeads, numDecoderLayers, pDropout))
  {
    add(encoder_);
    add(decoder_);
  };



  //std::vector<Variable> forward(
      //const Variable& src,
      //const Variable& query_embed) {
  //}
std::vector<Variable> forward(const std::vector<Variable>& input) override {
    assert(input.size() > 1);
    auto src = input[0];
    auto tgt = input[1];
    assert(src.dims(2) == tgt.dims(0));
    int B = src.dims(3);
    int C = src.dims(2);
    
    // Reshape from W X H X C X B to C X B X WH
    src = dataset::flatten(src, 0, 1);
    src = reorder(src, 1, 2, 0);

    tgt = moddims(tgt, { tgt.dims(0), 1, tgt.dims(1) });
    tgt = tile(tgt, {1, B, 1});
    auto memory = encoder_->forward({src});
    std::vector<Variable> decoderInputs = { tgt, memory[0] };
    auto hs = decoder_->forward(decoderInputs)[0];
    return { reorder(hs, 0, 2, 1) };
  }

  std::string prettyString() const override {
    return "Transformer";
  }

private:
  std::shared_ptr<TransformerEncoder> encoder_;
  std::shared_ptr<TransformerDecoder> decoder_;
};


} // namespace cv
} // namespace fl
