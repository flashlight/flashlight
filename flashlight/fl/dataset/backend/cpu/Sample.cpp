#include <string.h>

#include <flashlight/fl/dataset/Sample.h>

namespace fl {
Sample::Sample() : status_(Status::SELF_MANAGED) {}

Sample::Sample(const af::array& array)
    : array_(array), status_(Status::MANAGED_BY_AF) {}

void Sample::copyFromHost(
    void* hostPtr,
    af::dim4 dims,
    size_t sizeInBytes,
    af::dtype type) {
  array_ = af::array();
  dims_ = dims;
  status_ = Status::SELF_MANAGED;
  type_ = type;
  sizeInBytes_ = sizeInBytes;

  if (sizeInBytes > sysMemSize_) {
    if (sysMemPtr_) {
      free(sysMemPtr_);
    }
    sysMemPtr_ = malloc(sizeInBytes);
    sysMemSize_ = sizeInBytes;
  }

  memcpy(sysMemPtr_, hostPtr, sizeInBytes);
}

af::array&& Sample::array() {
  if (status_ == Status::RETIRED) {
    throw std::runtime_error("array() is called after the content is moved.");
  }

  if (status_ == Status::SELF_MANAGED) {
    if (type_ == u8) {
      array_ = af::array(dims_, (unsigned char*)sysMemPtr_);
    } else {
      std::string message = "Copy from arrayfire type, " +
          std::to_string((uint)type_) + " is not implemented yet.";
      throw std::runtime_error(message);
    }
  }

  status_ = Status::RETIRED;
  return std::move(array_);
}

af::dim4 Sample::dims() const {
  if (status_ == Status::MANAGED_BY_AF) {
    return array_.dims();
  }
  return dims_;
}

dim_t Sample::dims(unsigned int dim) const {
  if (status_ == Status::MANAGED_BY_AF) {
    return array_.dims(dim);
  }
  return dims_[dim];
}

dim_t Sample::elements() const {
  if (status_ == Status::MANAGED_BY_AF) {
    return array_.elements();
  }
  return dims_[0] * dims_[1] * dims_[2] * dims_[3];
}

void Sample::setArray(af::array& array) {
  status_ = Status::MANAGED_BY_AF;
  array_ = array;
}

void Sample::toDeviceAsync() {
  return;
}

af::dtype Sample::type() const {
  if (status_ == Status::MANAGED_BY_AF) {
    return array_.type();
  }
  return type_;
}

Sample::~Sample() {
  if (sysMemPtr_) {
    free(sysMemPtr_);
  }
}
} // namespace fl
