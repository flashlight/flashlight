/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/experimental/memory/AllocationLog.h"

#include <sstream>
#include <stdexcept>

#include "flashlight/fl/common/Logging.h"

namespace fl {

AllocationEvent::AllocationEvent(
    Type type,
    void* ptr,
    size_t sizeRequested,
    size_t sizeAllocated,
    size_t number)
    : type_(type),
      ptr_(ptr),
      sizeRequested_(sizeRequested),
      sizeAllocated_(sizeAllocated),
      number_(number) {}

AllocationEvent AllocationEvent::allocateNative(
    size_t sizeRequested,
    size_t sizeAllocated,
    void* ptr) {
  return AllocationEvent(
      AllocationEvent::Type::ALLOCATE_NATIVE,
      ptr,
      sizeRequested,
      sizeAllocated,
      0);
}

AllocationEvent AllocationEvent::allocateCache(
    size_t sizeRequested,
    size_t sizeAllocated,
    void* ptr) {
  return AllocationEvent(
      AllocationEvent::Type::ALLOCATE_CACHE,
      ptr,
      sizeRequested,
      sizeAllocated,
      0);
}

AllocationEvent AllocationEvent::freeNative(void* ptr) {
  return AllocationEvent(AllocationEvent::Type::FREE_NATIVE, ptr, 0, 0, 0);
}

AllocationEvent AllocationEvent::freeCache(void* ptr) {
  return AllocationEvent(AllocationEvent::Type::FREE_CACHE, ptr, 0, 0, 0);
}

AllocationEvent AllocationEvent::startEpoc(size_t number) {
  return AllocationEvent(
      AllocationEvent::Type::START_EPOC, nullptr, 0, 0, number);
}

AllocationEvent AllocationEvent::startBatch(size_t number) {
  return AllocationEvent(
      AllocationEvent::Type::START_BATCH, nullptr, 0, 0, number);
}

bool AllocationEvent::operator==(const AllocationEvent& other) const {
  return type_ == other.type_ && ptr_ == other.ptr_ &&
      sizeRequested_ == other.sizeRequested_ &&
      sizeAllocated_ == other.sizeAllocated_ && number_ == other.number_;
}

AllocationEvent AllocationEvent::fromCsvString(const std::string& csv) {
  // Sanitize
  constexpr size_t kMinimalSize = 5;
  if (csv.size() < kMinimalSize || csv[1] != ',') {
    throw std::invalid_argument(
        "AllocationEvent::fromCsvString(csv=" + csv +
        ") invalid csv value. Expected [event],[ptr],[size].");
  }
  AllocationEvent event(AllocationEvent::Type::UNINITIALIZED, nullptr, 0, 0, 0);

  switch (csv[0]) {
    case 'a':
      event.type_ = AllocationEvent::Type::ALLOCATE_NATIVE;
      break;
    case 'f':
      event.type_ = AllocationEvent::Type::FREE_NATIVE;
      break;
    case 'A':
      event.type_ = AllocationEvent::Type::ALLOCATE_CACHE;
      break;
    case 'F':
      event.type_ = AllocationEvent::Type::FREE_CACHE;
      break;
    case 'e':
      event.type_ = AllocationEvent::Type::START_EPOC;
      break;
    case 'b':
      event.type_ = AllocationEvent::Type::START_BATCH;
      break;
    default:
      throw std::invalid_argument(
          "AllocationEvent::fromCsvString(csv=" + csv +
          ") invalid event type.");
  };
  // Skip the first 2 chareters which are the type and a comma.
  std::istringstream values(csv.substr(2));
  std::string value;
  std::getline(values, value, ',');
  if (!value.empty()) {
    event.ptr_ = reinterpret_cast<void*>(std::stoul(value, nullptr, 16));
  }
  std::getline(values, value, ',');
  if (!value.empty()) {
    event.sizeRequested_ = std::stoul(value, nullptr, 10);
  }
  std::getline(values, value, ',');
  if (!value.empty()) {
    event.sizeAllocated_ = std::stoul(value, nullptr, 10);
  }
  std::getline(values, value, ',');
  if (!value.empty()) {
    event.number_ = std::stoul(value, nullptr, 10);
  }

  return event;
}
size_t sizeRequested_;
size_t sizeAllocated_;

std::string AllocationEvent::toCsvString() const {
  std::stringstream ss;
  switch (type_) {
    case AllocationEvent::Type::ALLOCATE_NATIVE:
      ss << "a," << std::hex << ptr_ << ',' << std::dec << sizeRequested_ << ','
         << sizeAllocated_ << ",0";
      break;
    case AllocationEvent::Type::FREE_NATIVE:
      ss << "f," << std::hex << ptr_ << ',' << "0,0,0,0";
      break;
    case AllocationEvent::Type::ALLOCATE_CACHE:
      ss << "A," << std::hex << ptr_ << ',' << std::dec << sizeRequested_ << ','
         << sizeAllocated_ << ",0";
      break;
    case AllocationEvent::Type::FREE_CACHE:
      ss << "F," << std::hex << ptr_ << ',' << "0,0,0,0";
      break;
    case AllocationEvent::Type::START_EPOC:
      ss << "e,0,0,0," << std::dec << number_;
      break;
    case AllocationEvent::Type::START_BATCH:
      ss << "b,0,0,0," << std::dec << number_;
      break;
    default:
      FL_LOG(fl::ERROR) << "AllocationEvent::toCsvString() type_="
                        << static_cast<int>(type_) << " is invalid value.";
      break;
  }
  return ss.str();
}

std::vector<AllocationEvent> LoadAllocationLog(std::istream& is) {
  FL_LOG(fl::INFO) << "LoadAllocationLog() loading...";
  std::vector<AllocationEvent> allocationLog;
  std::vector<std::string> errors;
  size_t cnt = 0;
  while (!is.eof() && is.good()) {
    std::string line;
    std::getline(is, line);
    // print progress bar
    ++cnt;
    if (!(cnt % 100000)) {
      std::cout << '.';
      std::cout.flush();
    }
    if (!line.empty()) {
      try {
        allocationLog.push_back(AllocationEvent::fromCsvString(line));
      } catch (std::exception& ex) {
        errors.push_back(ex.what());
      }
    }
  }

  std::stringstream ss;
  ss << "\nAllocation log is loaded with " << allocationLog.size()
     << " entries. ";
  if (!errors.empty()) {
    FL_LOG(fl::INFO) << ss.str();
  } else {
    ss << " Number of invalid log entries=" << errors.size() << '('
       << ((static_cast<double>(errors.size()) / allocationLog.size()) * 100.0)
       << "%) invalid log entries={";
    for (const std::string& err : errors) {
      ss << err << ", ";
    }
    ss << '}';
    FL_LOG(fl::ERROR) << ss.str();
  }
  return allocationLog;
}

void saveAllocationLog(
    std::ostream& os,
    const std::vector<AllocationEvent>& allocationLog) {
  for (const AllocationEvent& event : allocationLog) {
    os << event.toCsvString() << std::endl;
  }
  os.flush();
}

}; // namespace fl
