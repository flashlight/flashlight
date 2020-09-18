#pragma once

namespace fl {
namespace cv {
  /*

fl::Variable transformerInitLinear(int32_t inDim, int32_t outDim) {
  float std = std::sqrt(1.0 / float(inDim));
  return fl::uniform(outDim, inDim, -std, std);
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
        float pDropout=0.f,
    ) {
      wq_ = std::make_shared<Linear>(
        transformerInitLinear(modelDim, headDim * numHeads))
      wk_ = std::make_shared<Linear>(
        transformerInitLinear(modelDim, headDim * numHeads))
      wv_ = std::make_shared<Linear>(
        transformerInitLinear(modelDim, headDim * numHeads))
      wf_(std::make_shared<Linear>(
          transformerInitLinear(headDim * numHeads, modelDim))),
      add(wq_);
      add(wk_);
      add(wv_);
      add(wf_);
  };

  std::vector<Variable> forward(
      const Variable queries,
      const Variable keys,
      const Variable values) {
      int n = queries[0].dims(1), bsz = query[0].dims(2);
      double pDrop = train_ ? pDropout_ : 0.0;


      auto q = transpose((*wq_)(queries);
      auto k = transpose((*wk_)(keys);
      auto v = transpose((*wv_)(values);

      //Variable mask, posEmb;
      //posEmb = tile(params_[0], af::dim4(1, 1, nHeads_ * bsz));
      //if (useMask_ && input.back().dims(1) > 1) {
        //mask = getMask(n, input.size() == 2);
      //}

      //int offset = (input.size() == 1) ? 0 : input[0].dims(1);
      auto result = transformerMultiheadAttention(
        q, k, v, posEmb, mask, nHeads_, pDrop, offset);
      return { (*wf_)(transpose(result)) };
  }

  std::vector<Variable> forward(const std::vector<Variable>& input) override {
    return this->forward(input[0], input[1], input[2]);
  }


  private:
  std::shared_ptr<Linear> wq_;
  std::shared_ptr<Linear> wk_;
  std::shared_ptr<Linear> wv_;
  std::shared_ptr<Linear> wf_;
}

//Sequential mlp(int32_t input_dims,
    //int32_t hidden_dims,
    //int32_t output_dims) {
    //Sequentail output;
    //output.add(
        //Linear(
    //}

//class MLP : public Sequn {
  //MLP(int32_t input_dim,
//}

class TransformerBaseLayer : public Container {
 public:
  TransformerBaseLayer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout
      ) :
      self_attn_(std::make_shared<MultiHeadAttention>(modelDim, modelDim, nHeads, pDropout));
      w1_(std::make_shared<Linear>(transformerInitLinear(modelDim, mlpDim))),
      w2_(std::make_shared<Linear>(transformerInitLinear(mlpDim, modelDim))),
      norm1_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))),
      norm2_(std::make_shared<LayerNorm>(std::vector<int>({0, 3})))
      { };


  Variable mlp(const Variable& in) {
    float pDropout = train_ ? pDropout_ : 0.0;
    return (*w2_)(dropout(relu((*w1_)(input)), pDropout));
  }

  Variable withPosEmbed(const Variable& input, const Variable& pos) {
    if (pos.isEmpty()) {
      return input;
    }
    return input + pos;
  }

  Variable selfAttention(
      const Variable& input,
      const Variable &pos) {
    auto k = withPosEmbed(input, pos);
    auto q = withPosEmbed(input, pos);
    return self_atten_->forward(q, k, input);
  }

 private:
  std::shared_ptr<MultiHeadAttention> self_attn_;
  std::shared_ptr<Linear> w1_, w2_;
  std::shared_ptr<LayerNorm> norm1_, norm2_;
  TransformerBaseLayer();

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
    auto src2 = norm1_(src);
    src2 = this->selfAttention(src, fl::Variable());
    src = src + dropout(src2, pDropout_);
    src = norm1(src);
    src2 = mpl(src2);
    src = src + dropout(src2, pDropout_);
    src2 = norm2_(src2);
    return src2;
  }

};

class TransformerDecoderLayer : public TransformerBaseLayer {
 public:
  TransformerEncoderLayer(
      int32_t modelDim,
      int32_t headDim,
      int32_t mlpDim,
      int32_t nHeads,
      float pDropout) :
      TransformerBaseLayer(modelDim, headDim, mlpDim, nHeads, pDropout),
      self_encoder_attn_(
          std::make_shared<MultiHeadAttention>(modelDim, modelDim, nHeads, pDropout)
      );
      { };

  std::vector<Variable> forward(const std::vector<Variable>& input) override {
    auto src = input[0];
    auto src2 = norm1_(src);
    src2 = this->selfAttention(src2, src2);
    //src2 = self_attn_.forward(q, k, src2);
    src = src + dropout(src2, pDropout_);
    src = norm1(src);
    src2 = mpl(src2);
    src = src + dropout(src2, pDropout_);
    src2 = norm2_(src2);
    return src2;
  }

private:
  std::shared_ptr<MultiHeadAttention> encoder_attn_;
};

//class TransformerEncoderLayer : public TransformerBaseLayer {
//}

  /**
 * A module which implements a Transformer.
 *
 * For details, see [Vaswani et al
 * (2017)](https://arxiv.org/abs/1706.03762).
 *
 * This module also supports layer drop regularization, as introduced in
 * [Fan et al (2019)](https://arxiv.org/abs/1909.11556).
 *
 * Input dimension at forward is assumed to be CxTxBx1, where C is the
 * number of features, T the sequence length and B the batch size.
 *
 */
//class TransformerEncoderLayer : public Container {
 //public:
  //TransformerEncoderLayer(
      //int32_t modelDim,
      //int32_t headDim,
      //int32_t mlpDim,
      //int32_t nHeads,
      //int32_t bptt,
      //float pDropout,
      //float pLayerdrop,
      //bool useMask = false,
      //bool preLN = false) :
      //self_attn_(std::make_shared<MultiHeadAttention>(modelDim, modelDim, nHeads, pDropout));
      //w1_(std::make_shared<Linear>(transformerInitLinear(modelDim, mlpDim))),
      //w2_(std::make_shared<Linear>(transformerInitLinear(mlpDim, modelDim))),
      //norm1_(std::make_shared<LayerNorm>(std::vector<int>({0, 3}))),
      //norm2_(std::make_shared<LayerNorm>(std::vector<int>({0, 3})))
      //{
      //};

  //Variable withPosEmbed(const Variable& in) {
    //// TODO
    //return in;
  //}

  //std::vector<Variable> forward(const std::vector<Variable>& input) override {
    //auto src = input[0];
    //auto src2 = norm1_(src);
    //q =  withPosEmbed(q);
    //k = q;
    //src2 = self_attn_.forward(q, k, src2);
    //src = src + dropout(src2, pDropout_);
    //src = norm1(src);
    //src2 = mpl(src2);
    //src = src + dropout(src2, pDropout_);
    //src2 = norm2_(src2);
    //return src2;
  //}


  //Variable mlp(const Variable& in) {
    //float pDropout = train_ ? pDropout_ : 0.0;
    //return (*w2_)(dropout(relu((*w1_)(input)), pDropout));
  //}
  //std::string prettyString() const override;

 //private:
  //int32_t nHeads_;
  //int32_t bptt_;
  //double pDropout_;
  //double pLayerdrop_;
  //bool useMask_;
  //bool preLN_;
  //std::shared_ptr<MultiHeadAttention> self_attn_;
  //std::shared_ptr<Linear> w1_, w2_;
  //std::shared_ptr<LayerNorm> norm1_, norm2_;

  //Variable mlp(const Variable& input);
  //Variable getMask(int32_t n, bool cache = false);
  //Variable selfAttention(const std::vector<Variable>& input);

  //TransformerEncoderLayer();
//};

//CEREAL_REGISTER_TYPE(fl::cv::TransformerEncoderLayer);
//*

} // namespace cv
} // namespace fl
