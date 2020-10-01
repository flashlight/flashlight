/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace fl {

/**
 * Serializable abstraction of memory allocation/deallocation event.
 */
struct AllocationEvent {
  enum class Type {
    UNINITIALIZED,
    ALLOCATE_NATIVE,
    FREE_NATIVE,
    ALLOCATE_CACHE,
    FREE_CACHE,
    START_EPOC,
    START_BATCH,
  };
  Type type_;
  void* ptr_;
  size_t sizeRequested_;
  size_t sizeAllocated_;
  size_t number_; // used for epocs and batchs

  AllocationEvent(
      Type type,
      void* ptr,
      size_t sizeRequested,
      size_t sizeAllocated,
      size_t number);

  static AllocationEvent
  allocateNative(size_t sizeRequested, size_t sizeAllocated, void* ptr);
  static AllocationEvent
  allocateCache(size_t sizeRequested, size_t sizeAllocated, void* ptr);
  static AllocationEvent freeNative(void* ptr);
  static AllocationEvent freeCache(void* ptr);
  static AllocationEvent startEpoc(size_t number);
  static AllocationEvent startBatch(size_t number);

  bool operator==(const AllocationEvent& other) const;

  // Serialization
  std::string toCsvString() const;
  /**
   * Expected csv format:
   * [type],[ptr],[size]
   * type:  type should be 'a' for allocate or 'f' for free.
   * ptr:   the returned value from allocation or the supplied value for free.
   *        The pointer is formatted as an hexadecimal value.
   * size:  size to be allocated for event type='a'. In case of event type ='s'
   *        size may be the size of the freed object or zero. Size is formatted
   *        as a decimal value.
   *        This holds the number at case of batch or epoc.
   */
  static AllocationEvent fromCsvString(const std::string& csv);
};

/**
 * Serializes vector of AllocationEvent (a.k.a allocation log).
 */
void saveAllocationLog(
    std::ostream& os,
    const std::vector<AllocationEvent>& allocationLog);

/**
 * Deserializes vector of AllocationEvent (a.k.a allocation log).
 */
std::vector<AllocationEvent> LoadAllocationLog(std::istream& is);

} // namespace fl
