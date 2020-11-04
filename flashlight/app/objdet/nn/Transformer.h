#pragma once

#include "flashlight/fl/nn/nn.h"
#include "flashlight/app/objdet/dataset/BoxUtils.h"
#include "iostream"

#include <cassert>

// TODO check layer norm dimensions

namespace fl {
namespace app {
namespace objdet {

fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std);
}

//fl::Variable transformerRotate(const fl::Variable& input) {
  //auto data = input.array();
  //int d0 = data.dims(0);
  //int d1 = data.dims(1);
  //int d2 = data.dims(2);
  //int d3 = data.dims(3);
  //data = af::join(0, data, af::constant(0.0, d1, d1, d2, d3));
  //data = af::moddims(data, af::dim4((d0 + d1) * d1, 1, d2, d3));
  //data = data.rows(0, (d1 + d0 - 1) * d1 - 1);
  //data = af::moddims(data, af::dim4(d0 + d1 - 1, d1, d2, d3));
  //auto gradFunc = [d0, d1, d2, d3](
                      //std::vector<fl::Variable>& inputs,
                      //const fl::Variable& gradOutput) {
    //auto gradData = gradOutput.array();
    //gradData = af::moddims(gradData, af::dim4((d0 + d1 - 1) * d1, 1, d2, d3));
    //gradData = af::join(0, gradData, af::constant(0.0, d1, 1, d2, d3));
    //gradData = af::moddims(gradData, af::dim4(d0 + d1, d1, d2, d3));
    //gradData = gradData.rows(0, d0 - 1);
    //inputs[0].addGrad(fl::Variable(gradData, false));
  //};
  //return fl::Variable(data, {input}, gradFunc);
//}

// query [ E X B X L ]
// values and keys [ E X B X S ]
fl::Variable transformerMultiheadAttention(
    const fl::Variable& query,
    const fl::Variable& key,
    const fl::Variable& value,
    const fl::Variable& posEmb,
    const fl::Variable& keyPaddingMask,
    const int32_t nHead,
    const double pDropout
    ) {

  int32_t bsz = query.dims(1);
  int32_t modelDim = query.dims(0);
  int32_t headDim = modelDim / nHead;
  int32_t tgtLen = query.dims(2);
  int32_t srcLen = key.dims(2);

  auto q = moddims(query, af::dim4(headDim, nHead, bsz, tgtLen));
  auto v = moddims(value, af::dim4(headDim, nHead, bsz, srcLen));
  auto k = moddims(key, af::dim4(headDim, nHead, bsz, srcLen)); 
  // Reorder so that the "Sequence" is along the first dimension,
  // the embedding is along the zeroth dimension
  q = reorder(q, 0, 3, 1, 2);
  v = reorder(v, 0, 3, 1, 2);
  k = reorder(k, 0, 3, 1, 2);

  auto scores = matmulTN(q, k);
  scores = scores / std::sqrt(float(headDim));


  if(!keyPaddingMask.isempty()) {
    scores = scores + tileAs(moddims(log(keyPaddingMask), { 1, srcLen, 1, bsz }), scores);
  }

  auto attn = dropout(softmax(scores, 1), pDropout);
  auto result = matmulNT(attn, v);
  result = moddims(result, af::dim4(tgtLen, modelDim, bsz));
  result = reorder(result, 1, 2, 0);
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

  // queries [ E, N, L ], where L is target length, N is batch size.
  // keys / values  [ E, N, S ], where S is src length, N is batch size.
  // keyPaddingMask [ S, N ]
    std::vector<Variable> forward(
        const Variable queries,
        const Variable keys,
        const Variable values,
        const Variable keyPaddingMask
        ) {

      assert(queries.dims(0) == keys.dims(0));
      assert(queries.dims(0) == values.dims(0));
      assert(queries.dims(1) == keys.dims(1));
      assert(queries.dims(1) == values.dims(1));
      assert(values.dims(2) ==  keys.dims(2));


      // TODO Test for now
      //assert(!keyPaddingMask.isempty());
      if(!keyPaddingMask.isempty()) {
        assert(keyPaddingMask.dims(0) == keys.dims(2));
        assert(keyPaddingMask.dims(1) == keys.dims(1));
      }

      auto q = (*wq_)(queries);
      auto k = (*wk_)(keys);
      auto v = (*wv_)(values);
      auto posEmb = fl::Variable();
      //auto mask = fl::Variable();
      auto result = transformerMultiheadAttention(
          q, k, v, posEmb, keyPaddingMask, numHeads_, pDropout_);
      result = (*wf_)(result);
      assert(result.dims() == queries.dims());
      std::vector<Variable> results = { result };
      return results;
    }

  std::vector<Variable> forward(const std::vector<Variable>& input) override {
    assert(input.size() == 4);
    return this->forward(input[0], input[1], input[2], input[3]);
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
      self_attn_(std::make_shared<MultiheadAttention>(modelDim, modelDim / nHeads, nHeads, pDropout)),
      w1_(std::make_shared<Linear>(transformerInitLinear(modelDim, mlpDim))),
      w2_(std::make_shared<Linear>(transformerInitLinear(mlpDim, modelDim))),
      //norm1_(std::make_shared<LayerNorm>(std::vector<int>(1))),
      //norm2_(std::make_shared<LayerNorm>(std::vector<int>(1))),
      //norm1_(std::make_shared<LayerNorm>(0, 1e-3, true, modelDim)),
      //norm2_(std::make_shared<LayerNorm>(0, 1e-3, true, modelDim)),
      norm1_(std::make_shared<LayerNorm>(std::vector<int>{0}, 1e-5, true, modelDim)),
      norm2_(std::make_shared<LayerNorm>(std::vector<int>{0}, 1e-5, true, modelDim)),
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
      const Variable &pos,
      const Variable& keyPaddingMask = Variable()) {
    auto k = withPosEmbed(input, pos);
    auto q = withPosEmbed(input, pos);
    return self_attn_->forward(q, k, input, keyPaddingMask)[0];
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
    assert(input.size() == 3);
    auto src = input[0];
    auto mask = input[1];
    auto pos = input[2];
    assert(!mask.isempty());
    // Self Attention
    {
      //std::cout << "her1 " << src.dims() << std::endl;
      auto src2 = (*norm1_)(src);
      src2 = this->selfAttention(src2, pos, mask);
      src = src + dropout(src2, pDropout_);
    }
    // MLP
    {
      //std::cout << "her2 " << src.dims() << std::endl;
      auto src2 = norm2_->forward(src);
      src2 = mlp(src2);
      src = src + dropout(src2, pDropout_);
    }
    return { src, mask, pos }; 
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
        encoder_attn_(std::make_shared<MultiheadAttention>(modelDim, modelDim / nHeads, nHeads, pDropout)),
        //norm3_(std::make_shared<LayerNorm>(0, 1e-3, true, modelDim))
        norm3_(std::make_shared<LayerNorm>(std::vector<int>{0}, 1e-5, true, modelDim))
        { 
          add(encoder_attn_);
          add(norm3_);
        };

  std::vector<Variable> forward(const std::vector<Variable>& input) override {
      assert(input.size() == 5);
      auto tgt = input[0];
      auto memory = input[1];
      auto pos = (input.size() > 2) ? input[2] : Variable();
      auto queryPos = (input.size() > 3) ? input[3] : Variable();
      auto memoryKeyPaddingMask = input[4];
    // Self attention
    {
      ////std::cout << " dec " << tgt.dims() << std::endl;
      auto tgt2 = (*norm1_)(tgt);
      tgt2 = this->selfAttention(tgt2, queryPos);
      tgt = tgt + dropout(tgt2, pDropout_);
    }
    // Encoder-decoder attention
    {
      //std::cout << " dec2 " << tgt.dims() << std::endl;
      auto tgt2 = (*norm2_)(tgt);
      tgt2 = encoder_attn_->forward({
          this->withPosEmbed(tgt2, queryPos), // queries
          this->withPosEmbed(memory, pos), // keys
          memory, // values
          memoryKeyPaddingMask // mask
          })[0];
      // TODO fix norms
      tgt = tgt + dropout(tgt2, pDropout_);
    }
    // MLP
    {
      //std::cout << " dec3 " << tgt.dims() << std::endl;
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
      //add(LayerNorm(0, 1e-3, true, modelDim));
      add(LayerNorm(std::vector<int>{0}, 1e-5, true, modelDim));
    }

    std::vector<Variable> forward(const std::vector<Variable>& input) override {
      assert(input.size() == 5);
      auto tgt = input[0];
      auto memory = input[1];
      auto pos = (input.size() > 2) ? input[2] : Variable();
      auto query_pos = (input.size() > 3) ? input[3] : Variable();
      auto mask = (input.size() > 4) ? input[4] : Variable();

      fl::Variable output = tgt;
      auto mods = modules();
      for(int i = 0; i < mods.size() - 1; i++) {
        output = mods[i]->forward({output, memory, pos, query_pos, mask})[0];
      }
      output = mods.back()->forward({output})[0];
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
      float pDropout) 
      :
        norm_(std::make_shared<LayerNorm>(std::vector<int>{0}, 1e-5, true, modelDim))
    {
      for(int i = 0; i < layers; i++) {
        add(TransformerEncoderLayer(modelDim, headDim, mlpDim, nHeads, pDropout));
      }
      add(norm_);
    }

    std::vector<Variable> forward(const std::vector<Variable>& input) override {
      std::vector<Variable> output = input;
      auto mods = modules();
      for(int i = 0; i < mods.size() - 1; i++) {
        output = mods[i]->forward(output);
      }
      return { norm_->forward(output[0]) };
    }

    std::string prettyString() const override {
      return "TransformerDecoder";
    }
private:
    std::shared_ptr<LayerNorm> norm_;
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


std::vector<Variable> forward(
    Variable src,
    Variable mask,
    Variable queryEmbed,
    Variable posEmbed) {
      assert(src.dims(2) == queryEmbed.dims(0));

      int B = src.dims(3);
      // Reshape from [ W X H X C X B ] to [ WH X C X B ]
      src = flatten(src, 0, 1);
      // Flatten to C x B x WH
      src = reorder(src, 1, 2, 0);

      posEmbed = flatten(posEmbed, 0, 1);
      posEmbed = reorder(posEmbed, 1, 2, 0);

      mask = flatten(mask, 0, 2);

      // TODO Should we ever not pass positional encodings to each layer?
      // https://github.com/fairinternal/detr/blob/master/models/transformer.py#L56

      // Tile object queries for each batch
      af::dim4 unsqueeze = { queryEmbed.dims(0), 1, queryEmbed.dims(1) };
      queryEmbed = moddims(queryEmbed, unsqueeze);
      queryEmbed = tile(queryEmbed, {1, B, 1});
      assert(queryEmbed.dims(1) == src.dims(1));
      assert(queryEmbed.dims(0) == src.dims(0));

      auto tgt = fl::Variable(af::constant(0, queryEmbed.dims()), false);
      //auto tgt = queryEmbed;

      auto memory = encoder_->forward({
          src,
          mask,
          posEmbed
      });
      auto hs = decoder_->forward({
          tgt,
          memory[0],
          posEmbed,
          queryEmbed,
          mask})[0];
      return { reorder(hs, 0, 2, 1) };
  }

std::vector<Variable> forward(const std::vector<Variable>& input) override {
    //assert(input.size() > 1);
    assert(input.size() > 3);
    auto src = input[0];
    auto mask = (input.size() > 1) ? input[1] : Variable();
    auto query_embed = (input.size() > 2) ? input[2] : Variable();
    auto pos_embed = (input.size() > 3) ? input[3] : Variable();
    return forward(src, mask, query_embed, pos_embed);
  }

  std::string prettyString() const override {
    return "Transformer";
  }

private:
  std::shared_ptr<TransformerEncoder> encoder_;
  std::shared_ptr<TransformerDecoder> decoder_;
};


} // namespace objdet
} // namespace app
} // namespace fl
