#include "vision/criterion/Hungarian.h"
#include "vision/criterion/SetCriterion.h"

#include "flashlight/autograd/Variable.h"
#include "flashlight/flashlight.h"

#include <vector>

#include <gtest/gtest.h>

using namespace fl;
using namespace fl::cv;

TEST(HungarianMatcher, Test1) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 2;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 1;
  std::vector<float> predBoxesVec = {
    2, 2, 3, 3,
    1, 1, 2, 2
  };

  std::vector<float> targetBoxesVec = {
    1, 1, 2, 2,
    2, 2, 3, 3
  };

  std::vector<float> predLogitsVec = {
    1, 2,
    2, 1
  };

  std::vector<float> targetClassVec = {
    0, 1
  };
  std::vector<float> expRowIds = {
    0, 1
  };
  std::vector<float> colRowIds = {
    1, 0
  };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto matcher = HungarianMatcher(1, 1, 1);
  auto costs = matcher.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  auto rowIds = costs[0];
  auto colIds = costs[1];
  af_print(colIds);
  for(int i = 0; i < NUM_PREDS; i++) {
    EXPECT_EQ(colIds(i).scalar<int>(), colRowIds[i]);
  }
}

TEST(HungarianMatcher, Test2) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 1;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 1;
  std::vector<float> predBoxesVec = {
    2, 2, 3, 3,
    1, 1, 2, 2
  };

  std::vector<float> targetBoxesVec = { 1, 1, 2, 2, };
  std::vector<float> predLogitsVec = { 2, 1 };
  std::vector<float> targetClassVec = { 0 };
  std::vector<float> expRowIds = { 0 };
  std::vector<float> colRowIds = { 1 };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto matcher = HungarianMatcher(1, 1, 1);
  auto costs = matcher.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  auto rowIds = costs[0];
  auto colIds = costs[1];
  for(int i = 0; i < NUM_TARGETS; i++) {
    EXPECT_EQ(colIds(i).scalar<int>(), colRowIds[i]);
  }
  af_print(colIds);
  af_print(rowIds);
}

TEST(HungarianMatcher, Test3) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 1;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 1;
  std::vector<float> predBoxesVec = {
    1, 1, 2, 2,
    2, 2, 3, 3
  };

  std::vector<float> targetBoxesVec = { 1, 1, 2, 2, };
  std::vector<float> predLogitsVec = { 2, 1 };
  std::vector<float> targetClassVec = { 0 };
  std::vector<float> expRowIds = { 0 };
  std::vector<float> colRowIds = { 0 };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto matcher = HungarianMatcher(1, 1, 1);
  auto costs = matcher.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  auto rowIds = costs[0];
  auto colIds = costs[1];
  for(int i = 0; i < NUM_TARGETS; i++) {
    EXPECT_EQ(colIds(i).scalar<int>(), colRowIds[i]);
  }
  af_print(colIds);
  af_print(rowIds);
}

TEST(HungarianMatcher, TestBatching) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 2;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 2;
  std::vector<float> predBoxesVec = {
    1, 1, 2, 2,
    2, 2, 3, 3,
    2, 2, 3, 3,
    1, 1, 2, 2
  };

  std::vector<float> targetBoxesVec = { 
    1, 1, 2, 2, 
    2, 2, 3, 3,
    1, 1, 2, 2, 
    2, 2, 3, 3,
  };
  std::vector<float> predLogitsVec = { 2, 1, 2, 1 };
  std::vector<float> targetClassVec = { 0, 0, 0, 0 };
  std::vector<float> expRowIds = { 0, 1, 0, 1 };
  std::vector<float> colRowIds = { 0, 1, 1, 0 };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto matcher = HungarianMatcher(1, 1, 1);
  auto costs = matcher.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  for(int b = 0; b < NUM_BATCHES; b++) {
    auto colIds = costs[b * 2 + 1];
    af_print(colIds);
    for(int i = 0; i < NUM_TARGETS; i++) {
      EXPECT_EQ(colIds(i).scalar<int>(), colRowIds[i + b * NUM_TARGETS]);
    }
  }
}

// TODO make work for batching / dimension interpolation
// TODO make work with leaving out arguments
af::array lookup(const af::array& in, af::array idx0, af::array idx1, af::array idx2, af::array idx3) {
  auto inDims = in.dims();
  idx0 = (idx0.isempty()) ? af::iota({inDims[0]}) : idx0;
  idx1 = (idx1.isempty()) ? af::iota({1, inDims[1]}) : idx1;
  idx2 = (idx2.isempty()) ? af::iota({1, 1, inDims[2]}) : idx2;
  idx3 = (idx3.isempty()) ? af::iota({1, 1, 1,  inDims[3]}) : idx3;
  af::dim4 stride = { 1, inDims[0], inDims[0] * inDims[1], inDims[0] * inDims[1] * inDims[2] };
  af::array linearIndices = batchFunc(idx0 * stride[0], idx1 * stride[1], af::operator+);
  linearIndices = batchFunc(linearIndices, idx2 * stride[2], af::operator+);
  linearIndices = batchFunc(linearIndices, idx3 * stride[3], af::operator+);
  af::array output = af::constant(0.0, linearIndices.dims());
  output(af::seq(linearIndices.elements())) = in(linearIndices);
  return output;
}

fl::Variable lookup(const fl::Variable& in, af::array idx0, af::array idx1, af::array idx2, af::array idx3) {
  auto idims = in.dims();
  auto result = lookup(in.array(), idx0, idx1, idx2, idx3);
  auto gradFunction = [idx0, idx1, idx2, idx3, idims](std::vector<Variable>& inputs,
                                              const Variable& grad_output) {
        af_print(grad_output.array());
        if (!inputs[0].isGradAvailable()) {
          auto grad = af::constant(0.0, idims);
          inputs[0].addGrad(Variable(grad, false));
        }
        auto grad = fl::Variable(af::constant(0, idims), false);
        auto inDims = idims;
        auto idx0_ = (idx0.isempty()) ? af::iota({idims[0]}) : idx0;
        auto idx1_ = (idx1.isempty()) ? af::iota({1, idims[1]}) : idx1;
        auto idx2_ = (idx2.isempty()) ? af::iota({1, 1, idims[2]}) : idx2;
        auto idx3_ = (idx3.isempty()) ? af::iota({1, 1, 1,  idims[3]}) : idx3;
        af::dim4 stride = { 1, idims[0], idims[0] * idims[1], idims[0] * idims[1] * idims[2] };
        af::array linearIndices = batchFunc(idx0_ * stride[0], idx1_ * stride[1], af::operator+);
        linearIndices = batchFunc(linearIndices, idx2_ * stride[2], af::operator+);
        linearIndices = batchFunc(linearIndices, idx3_ * stride[3], af::operator+);
        // TODO Can parallize this if needed but does not work for duplicate keys
        for(int i = 0; i < linearIndices.elements(); i++) {
          af::array index = linearIndices(i);
          grad.array()(index) += grad_output.array()(i);
        }
        inputs[0].addGrad(grad);
  };
  return fl::Variable(result, { in.withoutData() }, gradFunction);
}

af::array lookup(const af::array& in, af::array idx, int dim=0) {

}

TEST(HungarianMatcher, TestArrayfire) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 2;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 2;
  std::vector<float> predBoxesVec = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };

  std::vector<float> targetBoxesVec = { 
    1, 1, 2, 2, 
    2, 2, 3, 3,
    1, 1, 2, 2, 
    2, 2, 3, 3,
  };
  std::vector<float> predLogitsVec = { 2, 1, 2, 1 };
  std::vector<float> targetClassVec = { 0, 0, 0, 0 };
  std::vector<int> expRowIds = { 0, 1, 0, 1 };
  std::vector<int> colRowIds = { 1, 0 };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());

  std::vector<int> colRowIds = { 1, 0 };
  auto colIdxs = af::array(1, 1, 2, colRowIds.data());

  auto input = fl::Variable(predBoxes, true);
  auto looked = lookup(input, af::array(), colIdxs, af::array(), af::array());
  auto loss = looked * looked;
  af_print(loss.array())
  loss.backward();
  af_print(loss.array())
  af_print(input.grad().array());
  //af_print(predBoxes);
  //af_print(idxs(0, af::span));
  //af_print(af::lookup(predBoxes, idxs(0, af::span), 1));
  //af_print(af::lookup(predBoxes, idxs(0, af::span), 0));
  //af_print(af::lookup(predBoxes, idxs(af::span, 0), 1));
  //af_print(idxs(af::span, 0));
  //af_print(af::lookup(predBoxes, idxs(af::span, 0), 0));
  //af_print(af::lookup(predBoxes, idxs(af::span, 0), 1));
  //gfor(af::seq i, NUM_BATCHES) {
    //af::array index = idxs(af::span, i);
    //af_print(index);
    //af_print(predBoxes(af::span, index, i));
    //af_print(predBoxes(af::span, index, af::span));
    //predBoxes(af::span, af::span, i) = predBoxes(af::span, index, i);
  //}
  //gfor(af::seq i, NUM_BATCHES) {
    //af_print(i);
    //af_print(idxs(i));
    //af_print(af::tile(idxs, 2)(i));
    //predBoxes(af::span, af::span, i) = af::lookup(predBoxes(af::span, af::span, i), idxs(i, af::span), 1);
  //}
  //af::seq batches = af::seq(0, NUM_BATCHES);
  //af::seq preds = af::seq(0, NUM_PREDS);
  //af::seq first = af::seq(0, 3);
  //predBoxes(af::span, af::span, batches) = af::lookup(predBoxes(af::span, af::span, batches), idxs(af::span, af::span, batches), 1);
  //af_print(af::index(idxs(af::span, af::span, batches)));
  //af_print(predBoxes);
  //af_print(af::index_gen(idxs, af::seq(0, 1)));
  //af_print(predBoxes(first, af::index(idxs(af::span, 0, 1)), 1));
  //predBoxes(af::span, af::span, batches) = predBoxes(af::span, af::index(idxs(af::span, 1, af::span)), af::span);
  //af_print(predBoxes);
  //af_print(af::lookup(predBoxes(af::span, af::span, 1), idxs(af::span, 0), 1));
  //for(int b = 0; b < NUM_BATCHES; b++) {
    //af::array test = idxs.col(b);
    //af_print(test);
    //predBoxes.slice(b) = af::lookup(predBoxes.slice(b), idxs.col(b), 1);
  //}
  //for(int b = 0; b < NUM_BATCHES; b++) {
    ////af::array test = idxs.col(b);
    ////af_print(test);
    //predBoxes.slice(b) = af::lookup(predBoxes.slice(b), idxs.col(b), 1);
  //}
  af_print(predBoxes);

}

TEST(SetCriterion, Test1) {
  const int NUM_CLASSES = 2;
  const int NUM_TARGETS = 2;
  const int NUM_PREDS = 2;
  const int NUM_BATCHES = 2;
  std::vector<float> predBoxesVec = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 10, 11, 12,
    13, 14, 15, 16
  };

  std::vector<float> targetBoxesVec = { 
    1, 1, 2, 2, 
    2, 2, 3, 3,
    1, 1, 2, 2, 
    2, 2, 3, 3,
  };
  std::vector<float> predLogitsVec = { 2, 1, 2, 1 };
  std::vector<float> targetClassVec = { 0, 0, 0, 0 };
  std::vector<int> expRowIds = { 0, 1, 0, 1 };
  std::vector<int> colRowIds = { 1, 0 };
  auto predBoxes = af::array(4, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto predLogits = af::array(NUM_CLASSES, NUM_PREDS, NUM_BATCHES, predBoxesVec.data());
  auto targetBoxes = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());
  auto targetClasses = af::array(4, NUM_TARGETS, NUM_BATCHES, targetBoxesVec.data());

  std::vector<int> colRowIds = { 1, 0 };
  auto colIdxs = af::array(1, 1, 2, colRowIds.data());

  auto input = fl::Variable(predBoxes, true);
  auto looked = lookup(input, af::array(), colIdxs, af::array(), af::array());
  auto loss = looked * looked;
  af_print(loss.array())
  loss.backward();
  af_print(loss.array())
  af_print(input.grad().array());
  //af_print(predBoxes);
  //af_print(idxs(0, af::span));
  //af_print(af::lookup(predBoxes, idxs(0, af::span), 1));
  //af_print(af::lookup(predBoxes, idxs(0, af::span), 0));
  //af_print(af::lookup(predBoxes, idxs(af::span, 0), 1));
  //af_print(idxs(af::span, 0));
  //af_print(af::lookup(predBoxes, idxs(af::span, 0), 0));
  //af_print(af::lookup(predBoxes, idxs(af::span, 0), 1));
  //gfor(af::seq i, NUM_BATCHES) {
    //af::array index = idxs(af::span, i);
    //af_print(index);
    //af_print(predBoxes(af::span, index, i));
    //af_print(predBoxes(af::span, index, af::span));
    //predBoxes(af::span, af::span, i) = predBoxes(af::span, index, i);
  //}
  //gfor(af::seq i, NUM_BATCHES) {
    //af_print(i);
    //af_print(idxs(i));
    //af_print(af::tile(idxs, 2)(i));
    //predBoxes(af::span, af::span, i) = af::lookup(predBoxes(af::span, af::span, i), idxs(i, af::span), 1);
  //}
  //af::seq batches = af::seq(0, NUM_BATCHES);
  //af::seq preds = af::seq(0, NUM_PREDS);
  //af::seq first = af::seq(0, 3);
  //predBoxes(af::span, af::span, batches) = af::lookup(predBoxes(af::span, af::span, batches), idxs(af::span, af::span, batches), 1);
  //af_print(af::index(idxs(af::span, af::span, batches)));
  //af_print(predBoxes);
  //af_print(af::index_gen(idxs, af::seq(0, 1)));
  //af_print(predBoxes(first, af::index(idxs(af::span, 0, 1)), 1));
  //predBoxes(af::span, af::span, batches) = predBoxes(af::span, af::index(idxs(af::span, 1, af::span)), af::span);
  //af_print(predBoxes);
  //af_print(af::lookup(predBoxes(af::span, af::span, 1), idxs(af::span, 0), 1));
  //for(int b = 0; b < NUM_BATCHES; b++) {
    //af::array test = idxs.col(b);
    //af_print(test);
    //predBoxes.slice(b) = af::lookup(predBoxes.slice(b), idxs.col(b), 1);
  //}
  //for(int b = 0; b < NUM_BATCHES; b++) {
    ////af::array test = idxs.col(b);
    ////af_print(test);
    //predBoxes.slice(b) = af::lookup(predBoxes.slice(b), idxs.col(b), 1);
  //}
  af_print(predBoxes);

}
