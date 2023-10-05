/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <af/algorithm.h>
#include <af/array.h>
#include <af/statistics.h>

#include <variant>

#include "flashlight/fl/runtime/Stream.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorAdapter.h"

namespace fl {

class ArrayFireBackend;

/**
 * Tensor adapter for the ArrayFire tensor library. Maps operations expressed in
 * Flashlight Tensors to ArrayFire.
 */
class ArrayFireTensor : public TensorAdapterBase {
  // A pointer to the internal ArrayFire array. Shared amongst tensors that are
  // shallow-copied.
  std::shared_ptr<af::array> arrayHandle_;

  // Indices in the event that this tensor is about to be indexed. Cleared the
  // next time this array handle is acquired. See getHandle().
  std::optional<std::vector<af::index>> indices_;
  // Need to maintain the types of each index, as ArrayFire doesn't distinguish
  // between an integer index literal and an af::seq of size one; both have
  // slightly different behavior with fl::Tensor
  std::optional<std::vector<detail::IndexType>> indexTypes_;
  // To be visited when this tensor is to be indexed. Indexes the underlying
  // af::array, and returns the proxy to be used as a temporary lvalue.
  struct IndexedArrayComponent {
    explicit IndexedArrayComponent(const bool _isFlat = false);
    af::array::array_proxy get(const ArrayFireTensor& inst);
    bool isFlat;
  };
  // To be visited when this tensor is holding an array without needing
  // indexing. Passthrough - returns the array directly.
  struct ArrayComponent {
    af::array& get(const ArrayFireTensor& inst);
  };
  // An interface to visit when getting an array handle. Indexes lazily
  // because we can't store an af::array::proxy as an lvalue. See getHandle().
  std::variant<ArrayComponent, IndexedArrayComponent> handle_{ArrayComponent()};

  /**
   * Constructs an ArrayFireTensor that will be lazily indexed.
   *
   * This constructor is for internal use only. Because af::array::array_proxy
   * objects don't work properly as lvalues, they need to be used as temporary
   * lvalues when doing in-place assignment. As such, Tensors are lazily-indexed
   * if operators that might need to operate on array proxies are called. This
   * ctor sets up that lazy indexing.
   *
   * Whenever these ArrayFireTensors are mutated, ArrayFireTensor::getHandle()
   * is called, which performs indexing if needed and upcasts the array_proxy to
   * a full af::array on which operations can be performed.
   *
   * @param[in] handle a pointer to the ArrayFire array
   * @param[in] indices a vector of ArrayFire indices to lazily index.
   * @param[in] indexTypes a vector of index types to lazily index. Needed to
   * determine singleton dimension condensation
   * @param[in] isFlat if the indexing op is flat (condense all dims)
   */
  ArrayFireTensor(
      std::shared_ptr<af::array> handle,
      std::vector<af::index>&& afIndices,
      std::vector<detail::IndexType>&& indexTypes,
      const unsigned numDims,
      const bool isFlat);

  /**
   * Construct an ArrayFireTensor from an ArrayFire array handle without copying
   * the handle. Used for creating guaranteed-shallow copies.
   */
  explicit ArrayFireTensor(std::shared_ptr<af::array> arr, unsigned numDims);

  /*
   * A Flashlight Shape that mirrors ArrayFire dims.
   *
   * NOTE: this shape is only updated on calls to ArrayFireTensor::shape()
   * so as to satisfy API requirements as per returning a const reference.
   * af::array::dims() should be used for internal computation where
   * shape/dimensions are needed.
   *
   * The default shape is the empty Tensor 0.
   */
  Shape shape_;

  /*
   * The number of dimensions in this ArrayFire tensor that are "expected" per
   * interoperability with other tensors. Because ArrayFire doesn't distinguish
   * between singleton dimensions that are defaults and those that are
   * explicitly specified, this must be explicitly tracked.
   *
   * The fl::Tensor default Tensor shape is {0} - the default number of numDims
   * is thus 1. Scalars have numDims == 0;
   */
  unsigned numDims_{1};

 public:
 constexpr static TensorBackendType tensorBackendType = TensorBackendType::ArrayFire;

  /**
   * Constructs an ArrayFireTensor.
   *
   * Since af::arrays are refcounted, an instance of this class
   * can only be created using arrays that are moved therein.
   *
   * Tensor operations occurring directly on this tensor's underlying
   * af::array should not copy the array else take a performance penalty (via
   * an internal copy if refcount is > 1 in some cases).
   *
   * @param[in] array construct a tensor from an ArrayFire array rvalue
   * reference.
   */
  explicit ArrayFireTensor(af::array&& array, const unsigned numDims);

  /**
   * Default initialization - empty ArrayFire array and empty shape.
   */
  ArrayFireTensor();

  /**
   * Construct an ArrayFire tensor using some data.
   *
   * @param[in] shape the shape of the new tensor
   * @param[in] ptr the buffer containing underlying tensor data
   * @param[in] type the type of the new tensor
   * @param[in] memoryLocation the location of the buffer
   */
  ArrayFireTensor(
      const Shape& shape,
      fl::dtype type,
      const void* ptr,
      Location memoryLocation);

  ArrayFireTensor(
      const Dim nRows,
      const Dim nCols,
      const Tensor& values,
      const Tensor& rowIdx,
      const Tensor& colIdx,
      StorageType storageType);

  /**
   * Gets an ArrayFire Array from this impl.
   *
   * Throws if this tensor represents an array_proxy, since it precludes
   * promotion to an array.
   */
  const af::array& getHandle() const;

  /**
   * Gets an ArrayFire Array from this impl. If the underlying handle is an
   * array_proxy, may promote it to an af::array condense dimensions as needed,
   * replace the handle variant, and return a reference.
   */
  af::array& getHandle();

  ~ArrayFireTensor() override = default;
  unsigned numDims() const;
  // Used with the fl::Tensor copy constructor
  std::unique_ptr<TensorAdapterBase> clone() const override;
  TensorBackendType backendType() const override;
  TensorBackend& backend() const override;
  Tensor copy() override;
  Tensor shallowCopy() override;
  const Shape& shape() override;
  dtype type() override;
  bool isSparse() override;
  af::dtype afHandleType(); // for internal use only
  Location location() override;
  void scalar(void* out) override;
  void device(void** out) override;
  void host(void* out) override;
  void unlock() override;
  bool isLocked() override;
  bool isContiguous() override;
  Shape strides() override;
  const Stream& stream() const override;
  Tensor astype(const dtype type) override;
  Tensor index(const std::vector<Index>& indices) override;
  Tensor flatten() const override;
  Tensor flat(const Index& idx) const override;
  Tensor asContiguousTensor() override;
  void setContext(void* context) override; // noop
  void* getContext() override; // noop
  std::string toString() override;
  std::ostream& operator<<(std::ostream& ostr) override;

  /******************** Assignment Operators ********************/
#define ASSIGN_OP_TYPE(OP, TYPE) void OP(const TYPE& val) override;

  /**
   * When indexing ArrayFire arrays, their dimensions are condensed (i.e. {3, 4,
   * 5, 6}(fl::span, 1) --> {3, 5, 6} rather than {3, 1, 5, 6}) when arrays are
   * returned as lvalues. In the case of lvalue temporary af::array::array_proxy
   * objects that have in-place operations applied to them, one can't modify
   * their dimensions without upcasting them into an af::array, which breaks
   * in-place op logic.
   *
   * The only option is thus to modify the dimensions of the operand of the
   * inplace operation in order to make the shapes match, but this should only
   * be done if the shapes are actually compatible. This function performs that
   * op before in-place operations are applied.
   *
   * @param[in] operand the tensor operand
   * @param[in] newNumDims the number of dims of the resulting tensor
   */
  af::array adjustInPlaceOperandDims(const Tensor& operand);

#define ASSIGN_OP(OP)                 \
  ASSIGN_OP_TYPE(OP, Tensor);         \
  ASSIGN_OP_TYPE(OP, double);         \
  ASSIGN_OP_TYPE(OP, float);          \
  ASSIGN_OP_TYPE(OP, int);            \
  ASSIGN_OP_TYPE(OP, unsigned);       \
  ASSIGN_OP_TYPE(OP, bool);           \
  ASSIGN_OP_TYPE(OP, char);           \
  ASSIGN_OP_TYPE(OP, unsigned char);  \
  ASSIGN_OP_TYPE(OP, short);          \
  ASSIGN_OP_TYPE(OP, unsigned short); \
  ASSIGN_OP_TYPE(OP, long);           \
  ASSIGN_OP_TYPE(OP, unsigned long);  \
  ASSIGN_OP_TYPE(OP, long long);      \
  ASSIGN_OP_TYPE(OP, unsigned long long);

  ASSIGN_OP(assign); // =
  ASSIGN_OP(inPlaceAdd); // +=
  ASSIGN_OP(inPlaceSubtract); // -=
  ASSIGN_OP(inPlaceMultiply); // *=
  ASSIGN_OP(inPlaceDivide); // /=
#undef ASSIGN_OP_TYPE
#undef ASSIGN_OP
};

/**
 * Gets an af::array from a Tensor. If the Tensor is not ArrayFire-backed,
 * throws an exception
 *
 * @param[in] tensor the input tensor
 * @return the array underying the Tensor
 */
const af::array& toArray(const Tensor& tensor);
af::array& toArray(Tensor& tensor);

} // namespace fl
