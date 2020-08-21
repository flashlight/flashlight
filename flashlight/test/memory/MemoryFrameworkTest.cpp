/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <arrayfire.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flashlight/common/CppBackports.h"
#include "flashlight/memory/memory.h"

using namespace fl;
using ::testing::_;
using ::testing::Exactly;
using ::testing::Invoke;

namespace {

/**
 * An extremely basic memory manager with a basic caching mechanism for testing
 * purposes. Not thread safe or optimized.
 *
 * Memory management still has to operate properly for tests to run;
 * tests operate on real ArrayFire arrays and allocated memory, and not properly
 * defining test closures will result in ArrayFire being in a bad internal
 * state.
 */
class TestMemoryManager : public MemoryManagerAdapter {
 public:
  TestMemoryManager(
      std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface,
      std::ostream* logStream)
      : MemoryManagerAdapter(deviceInterface, logStream) {}

  void initialize() override {}

  void shutdown() override {}

  void* alloc(
      bool userLock,
      const unsigned ndims,
      dim_t* dims,
      const unsigned elSize) override {
    size_t size = elSize;
    for (unsigned i = 0; i < ndims; ++i) {
      size *= dims[i];
    }
    void* ptr = nullptr;

    if (size > 0) {
      if (lockedBytes >= maxBytes || totalBytes >= maxBuffers) {
        signalMemoryCleanup();
      }

      ptr = this->deviceInterface->nativeAlloc(size);
      lockedPtrToSizeMap[ptr] = size;
      totalBytes += size;
      totalBuffers++;

      // Simple implementation: treat user and AF allocations the same
      locked.insert(ptr);
      lockedBytes += size;

      lastDims = af::dim4(ndims, dims);
    }
    return ptr;
  }

  size_t allocated(void* ptr) override {
    if (lockedPtrToSizeMap.find(ptr) == lockedPtrToSizeMap.end()) {
      return 0;
    } else {
      return lockedPtrToSizeMap[ptr];
    }
  }

  void unlock(void* ptr, bool userLock) override {
    if (!ptr) {
      return;
    }

    if (lockedPtrToSizeMap.find(ptr) == lockedPtrToSizeMap.end()) {
      return;
    }

    // For testing, treat user-allocated and AF-allocated memory identically
    if (locked.find(ptr) != locked.end()) {
      locked.erase(ptr);
      lockedBytes -= lockedPtrToSizeMap[ptr];
    }
  }

  void signalMemoryCleanup() override {
    // Free unlocked memory
    std::vector<void*> freed;
    for (auto& entry : lockedPtrToSizeMap) {
      if (!isUserLocked(entry.first)) {
        void* ptr = entry.first;
        this->deviceInterface->nativeFree(ptr);
        totalBytes -= lockedPtrToSizeMap[entry.first];
        freed.push_back(entry.first);
      }
    }
    for (auto ptr : freed) {
      lockedPtrToSizeMap.erase(ptr);
    }
  }

  void printInfo(const char* msg, const int device) override {}

  void userLock(const void* cPtr) override {
    void* ptr = const_cast<void*>(cPtr);
    if (locked.find(ptr) == locked.end()) {
      locked.insert(ptr);
      lockedBytes += lockedPtrToSizeMap[ptr];
    }
  }

  void userUnlock(const void* cPtr) override {
    void* ptr = const_cast<void*>(cPtr);
    unlock(ptr, /* user */ true);
    lockedBytes -= lockedPtrToSizeMap[ptr];
  }

  bool isUserLocked(const void* ptr) override {
    return locked.find(const_cast<void*>(ptr)) != locked.end();
  }

  float getMemoryPressure() override {
    if (lockedBytes > maxBytes || totalBuffers > maxBuffers) {
      return 1.0;
    } else {
      return 0.0;
    }
  }

  bool jitTreeExceedsMemoryPressure(size_t bytes) override {
    return 2 * bytes > lockedBytes;
  }

  void addMemoryManagement(int device) override {
    throw std::logic_error("Not implemented");
  }

  void removeMemoryManagement(int device) override {
    throw std::logic_error("Not implemented");
  }

  std::unordered_map<void*, size_t> lockedPtrToSizeMap;
  std::unordered_set<void*> locked;
  size_t totalBytes{0};
  size_t totalBuffers{0};
  size_t lockedBytes{0};
  size_t maxBuffers{64};
  size_t maxBytes{1024};
  // helps test dim_t* argument to alloc
  af::dim4 lastDims{0, 0, 0, 0};
};

/**
 * A mock of the simple test memory manager above.
 *
 * A mock class is created because all calls to mocked methods still need to be
 * dispatched into the main class; memory management still has to operate
 * properly for tests to run, as those tests operate on real ArrayFire arrays
 * and allocated memory.
 *
 * The below pattern uses a GMock delegated call to a real object - see
 * https://git.io/Jv59Z. All methods must be mocked if their calls are
 * dispatched to a derived real implementation even though those mocked methods
 * aren't directly tested.
 */
class MockTestMemoryManager : public TestMemoryManager {
 public:
  MockTestMemoryManager(
      std::shared_ptr<TestMemoryManager> real,
      std::shared_ptr<MemoryManagerDeviceInterface> deviceInterface,
      std::ostream* logStream)
      : TestMemoryManager(deviceInterface, logStream), real_(real) {
    ON_CALL(*this, initialize()).WillByDefault(Invoke([this]() {
      real_->initialize();
    }));
    ON_CALL(*this, shutdown()).WillByDefault(Invoke([this]() {
      real_->shutdown();
    }));
    ON_CALL(*this, alloc(_, _, _, _))
        .WillByDefault(Invoke([this](
                                  bool userLock,
                                  const unsigned ndims,
                                  dim_t* dims,
                                  const unsigned elSize) {
          return real_->alloc(userLock, ndims, dims, elSize);
        }));
    ON_CALL(*this, allocated(_)).WillByDefault(Invoke([this](void* ptr) {
      return real_->allocated(ptr);
    }));
    ON_CALL(*this, unlock(_, _))
        .WillByDefault(Invoke([this](void* ptr, bool userLock) {
          real_->unlock(ptr, userLock);
        }));
    ON_CALL(*this, signalMemoryCleanup()).WillByDefault(Invoke([this]() {
      real_->signalMemoryCleanup();
    }));
    ON_CALL(*this, printInfo(_, _))
        .WillByDefault(Invoke([this](const char* msg, const int device) {
          real_->printInfo(msg, device);
        }));
    ON_CALL(*this, userLock(_)).WillByDefault(Invoke([this](const void* cPtr) {
      real_->userLock(cPtr);
    }));
    ON_CALL(*this, userUnlock(_))
        .WillByDefault(
            Invoke([this](const void* cPtr) { real_->userUnlock(cPtr); }));
    ON_CALL(*this, isUserLocked(_))
        .WillByDefault(Invoke(
            [this](const void* cPtr) { return real_->isUserLocked(cPtr); }));
    ON_CALL(*this, getMemoryPressure()).WillByDefault(Invoke([this]() {
      return real_->getMemoryPressure();
    }));
    ON_CALL(*this, jitTreeExceedsMemoryPressure(_))
        .WillByDefault(Invoke([this](size_t bytes) {
          return real_->jitTreeExceedsMemoryPressure(bytes);
        }));
  }

  MOCK_METHOD0(initialize, void());
  MOCK_METHOD0(shutdown, void());
  MOCK_METHOD4(alloc, void*(bool, const unsigned, dim_t*, const unsigned));
  MOCK_METHOD1(allocated, size_t(void*));
  MOCK_METHOD2(unlock, void(void*, bool));
  MOCK_METHOD0(signalMemoryCleanup, void());
  MOCK_METHOD2(printInfo, void(const char*, const int));
  MOCK_METHOD1(userLock, void(const void*));
  MOCK_METHOD1(userUnlock, void(const void*));
  MOCK_METHOD1(isUserLocked, bool(const void*));
  MOCK_METHOD0(getMemoryPressure, float());
  MOCK_METHOD1(jitTreeExceedsMemoryPressure, bool(size_t));

 private:
  std::shared_ptr<TestMemoryManager> real_;
};

} // namespace

/**
 * Tests `MemoryManagerAdapter`, `MemoryManagerInstaller`, and
 * `MemoryManagerDeviceInterface`. By doing the following:
 * - Creating a custom memory manager (see `TestMemoryManager` above) and its
 *   mocked counterparts with a device interface and log stream.
 * - Setting the custom memory manager to be the active memory manager in
 *   ArrayFire using the `MemoryManagerInstaller`
 * - Performing memory allocations using af::alloc, which allocates user-locked
 *   memory, and ArrayFire functions which call the  `af::array` constructor
 *   (e.g. af::randu`).
 * - Mocking memory state operations that should be called via ArrayFire public
 *   API functions (memory cleanup, printing info, etc).
 * - Freeing allocated user memory and allowing `af::array`s to fall out of
 *   scope, facilitating unlocking
 * - Unsetting of the custom memory manager and restoration of the default
 *   memory manager
 */
TEST(MemoryFramework, AdapterInstallerDeviceInterfaceTest) {
  std::stringstream logStream;
  std::stringstream mockLogStream;
  {
    auto deviceInterface = std::make_shared<MemoryManagerDeviceInterface>();

    auto memoryManager =
        std::make_shared<TestMemoryManager>(deviceInterface, &logStream);
    auto mockMemoryManager = std::make_shared<MockTestMemoryManager>(
        memoryManager, deviceInterface, &mockLogStream);

    auto installer =
        cpp::make_unique<MemoryManagerInstaller>(mockMemoryManager);
    // initialize should only be called once while the custom memory
    // manager was set
    EXPECT_CALL(*mockMemoryManager, initialize()).Times(Exactly(1));
    installer->setAsMemoryManager();

    // flush the mock log stream every two lines
    mockMemoryManager->setLogFlushInterval(2);

    {
      // Do some sample allocations using `af::alloc` (which allocates
      // user-locked memory) and `af::randu`, which calls the `af::array`
      // constructor to allocate memory.
      EXPECT_CALL(*mockMemoryManager, alloc(/* user lock */ true, 1, _, 1))
          .Times(Exactly(1));
      size_t aSize = 8;
      void* a = af::alloc(aSize, af::dtype::f32);
      // Allocated memory should properly appear in our internal data structures
      // given correct passage of state
      EXPECT_EQ(memoryManager->lockedPtrToSizeMap.size(), 1);
      EXPECT_EQ(memoryManager->lockedPtrToSizeMap[a], aSize * sizeof(float));
      // Check that the dims for this array are correct. ArrayFire currently
      // passes (dims, 1, 1, 1) for all allocations.
      EXPECT_EQ(memoryManager->lastDims, af::dim4(aSize * sizeof(float)));

      // Check logs which should be flushed to our output stream after 2 ops
      std::string log1;
      std::getline(mockLogStream, log1);
      EXPECT_EQ(log1, "initialize ");
      std::string log2;
      std::getline(mockLogStream, log2);
      EXPECT_EQ(
          log2.substr(0, 14),
          "nativeAlloc " + std::to_string(aSize * sizeof(float)));

      // Buffer more logs in the default memory manager
      memoryManager->setLogFlushInterval(50);

      // Allocate an `af::array`, which won't be user locked, and has
      // information about array size passed to alloc
      dim_t bDim = 2;
      EXPECT_CALL(
          *mockMemoryManager,
          alloc(/* user lock */ false, 1, _, sizeof(float)));
      af::array b = af::randu({bDim, bDim});
      // Again, allocated should properly appear in our internal data structures
      // given correct passage of state
      EXPECT_EQ(memoryManager->totalBytes, aSize * sizeof(float) + b.bytes());
      EXPECT_EQ(memoryManager->totalBuffers, 2);
      // Our array is locked, but not user locked
      EXPECT_EQ(memoryManager->lockedBytes, aSize * sizeof(float) + b.bytes());
      EXPECT_EQ(memoryManager->locked.size(), 2);
      // Check that the dims for this array are correct
      EXPECT_EQ(memoryManager->lastDims, af::dim4(bDim * b.numdims()));

      // Free user-locked memory. Check that freeing memory properly calls
      // unlock with user-locked memory (since we used af::alloc)
      EXPECT_CALL(*mockMemoryManager, unlock(a, /* user lock */ true))
          .Times(Exactly(1));
      af::free(a);
      // Internal data structures should be updated accordingly to reflect
      // removal of a buffer
      EXPECT_EQ(memoryManager->totalBytes, aSize * sizeof(float) + b.bytes());
      EXPECT_EQ(memoryManager->totalBuffers, 2);
      EXPECT_EQ(memoryManager->lockedBytes, b.bytes());
      EXPECT_EQ(memoryManager->locked.size(), 1);

      // af::array b is out of scope, which is not user-locked memory
      EXPECT_CALL(*mockMemoryManager, unlock(_, /* user lock */ false))
          .Times(Exactly(1));
    }

    // Memory reset calls signalMemoryCleanup() and clears the map
    EXPECT_CALL(*mockMemoryManager, signalMemoryCleanup()).Times(Exactly(1));
    af::deviceGC();
    EXPECT_TRUE(memoryManager->lockedPtrToSizeMap.empty());

    // printInfo
    const std::string printInfoMsg = "testPrintInfo";
    int printInfoDeviceId = 0;
    EXPECT_CALL(
        *mockMemoryManager, printInfo(printInfoMsg.c_str(), printInfoDeviceId))
        .Times(Exactly(1));
    af::printMemInfo(printInfoMsg.c_str(), printInfoDeviceId);

    // all allocations are either freed or out of scope - check that the map is
    // empty
    EXPECT_TRUE(memoryManager->lockedPtrToSizeMap.empty());
    // reset to default memory manager
    // shutdown is called for each device with that current device set
    EXPECT_CALL(*mockMemoryManager, shutdown())
        .Times(Exactly(af::getDeviceCount()));
    MemoryManagerInstaller::unsetMemoryManager();
    // Test that unsetting a memory manager via the global singleton restores
    // the default ArrayFire memory manager
    auto* manager = MemoryManagerInstaller::currentlyInstalledMemoryManager();
    ASSERT_EQ(manager, nullptr);

    // Any allocations made should not call the custom memory manager since
    // we've called `MemoryManagerInstaller::unsetMemoryManager()` above, which
    // restores the default memory manager as the primary memory manager.
    EXPECT_CALL(*mockMemoryManager, alloc(_, _, _, _)).Times(Exactly(0));
    EXPECT_CALL(*mockMemoryManager, unlock(_, _)).Times(Exactly(0));
    dim_t cDim = 4;
    size_t pSize = 8;
    auto c = af::randu({cDim, cDim});
    void* p = af::alloc(pSize, af::dtype::f32);
    af::free(p);
  }
  // The custom memory is destroyed; check that the log stream, which is flushed
  // on destruction, contains the correct output
  std::vector<std::string> expectedLinePrefixes = {"initialize",
                                                   "nativeAlloc",
                                                   "alloc",
                                                   "nativeAlloc",
                                                   "alloc",
                                                   "unlock",
                                                   "unlock",
                                                   "signalMemoryCleanup",
                                                   "nativeFree",
                                                   "nativeFree",
                                                   "shutdown",
                                                   "shutdown"};
  size_t idx = 0;
  for (std::string line; std::getline(logStream, line);) {
    EXPECT_EQ(line.substr(0, line.find(" ")), expectedLinePrefixes[idx]);
    idx++;
  }

  // Test that normal allocations work now that the custom memory manager has
  // been destroyed and its function pointers and closures invalidated
  dim_t cDim = 4;
  size_t pSize = 8;
  auto c = af::randu({cDim, cDim});
  void* p = af::alloc(pSize, af::dtype::f32);
  af::free(p);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
