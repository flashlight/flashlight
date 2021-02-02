/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <random>
#include <vector>

#include <af/device.h>
#include <af/internal.h>
#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/fl/common/Init.h"
#include "flashlight/fl/memory/memory.h"

// class CachingMemoryManagerTest : public ::testing::Test {
//  protected:
//   virtual void SetUp() override {
//     deviceInterface_ = std::make_shared<fl::MemoryManagerDeviceInterface>();
//     adapter_ = std::make_shared<fl::CachingMemoryManager>(
//         af::getDeviceCount(), deviceInterface_);
//     installer_ = std::make_unique<fl::MemoryManagerInstaller>(adapter_);
//     installer_->setAsMemoryManager();
//   }

//   virtual void TearDown() override {
//     af_unset_memory_manager();
//   }

//   std::shared_ptr<fl::MemoryManagerDeviceInterface> deviceInterface_;
//   std::shared_ptr<fl::CachingMemoryManager> adapter_;
//   std::unique_ptr<fl::MemoryManagerInstaller> installer_;
// };

// TEST_F(CachingMemoryManagerTest, BasicOps) {
//   // This test checks if the basic math operations like additions,
//   // multiplication, division are performed correctly

//   const int nx = 8;
//   const int ny = 8;
//   af::array in1 = af::constant(2.0, nx, ny, u32);
//   af::array in2 = af::constant(3.0, nx, ny, u32);
//   ASSERT_TRUE(af::allTrue<bool>(in1 + in2 == 5));

//   // NOTE: includes JIT ops
//   af::array in3 = in1 * in2;
//   af::array in4 = in3 / in2;
//   af::array in5 = in4 * in2;
//   ASSERT_TRUE(af::allTrue<bool>(in3 == in5));
// }

// TEST_F(CachingMemoryManagerTest, DevicePtr) {
//   // This test checks whether device pointer API works for the arrays
//   // The CPU backend in AF allocates a buffer for empty arrays - see
//   // https://github.com/arrayfire/arrayfire/issues/3058. When this is fixed,
//   // this can be relaxed.
//   if (FL_BACKEND_CPU) {
//     GTEST_SKIP() << "ArrayFire CPU backend allocates buffers for empty arrays";
//   }

//   // Empty array
//   auto arr1 = af::array(0, 0, 0, 0, af::dtype::f32);
//   auto* ptr1 = arr1.device<float>();
//   ASSERT_EQ(ptr1, nullptr);
//   arr1.unlock();

//   // Non-Empty array
//   auto arr2 = af::array(10, 8, 9, 23, af::dtype::f32);
//   auto* ptr2 = arr2.device<float>();
//   ASSERT_NE(ptr2, nullptr);
//   arr2.unlock();
// }

// TEST_F(CachingMemoryManagerTest, IndexedDevice) {
//   // This test is checking to see if calling `.device()` will force copy to a
//   // new buffer unlike `getRawPtr()`. It is required to copy as the the memory
//   // manager releases the lock on the array after calling `.device()`
//   const int nx = 8;
//   const int ny = 8;

//   af::array in = af::randu(nx, ny);

//   std::vector<float> in1(in.elements());
//   in.host(in1.data());

//   int offx = nx / 4;
//   int offy = ny / 4;

//   in = in(af::seq(offx, offx + nx / 2 - 1), af::seq(offy, offy + ny / 2 - 1));

//   int nxo = (int)in.dims(0);
//   int nyo = (int)in.dims(1);

//   void* rawPtr = af::getRawPtr(in);
//   void* devPtr = in.device<float>();
//   ASSERT_NE(devPtr, rawPtr);
//   in.unlock();

//   std::vector<float> in2(in.elements());
//   in.host(in2.data());

//   for (int y = 0; y < nyo; y++) {
//     for (int x = 0; x < nxo; x++) {
//       ASSERT_EQ(in1[(offy + y) * nx + offx + x], in2[y * nxo + x]);
//     }
//   }
// }

// TEST_F(CachingMemoryManagerTest, LargeNumberOfAllocs) {
//   // This test performs stress test to allocate and free a large of number of
//   // array of variable sizes

//   std::stringstream log;
//   adapter_->setLogStream(&log);
//   adapter_->setLoggingEnabled(0x2);
//   af::array a;
//   for (int i = 0; i < 5000; ++i) {
//     auto dimsArr = (af::randu(4, af::dtype::s32)) % 100 + 100;
//     std::vector<int> dims(4);
//     dimsArr.as(af::dtype::s32).host(dims.data());
//     EXPECT_NO_THROW(a = af::array(dims[0], dims[1], dims[2], dims[3]));
//   }
// }

std::string formatMemory(size_t bytes) {
  const std::vector<std::string> units = {"B", "KiB", "MiB", "GiB", "TiB"};
  size_t unitId =
      bytes == 0 ? 0 : std::floor(std::log(bytes) / std::log(1024.0));
  unitId = std::min(unitId, units.size() - 1);
  std::string bytesStr = std::to_string(bytes / std::pow(1024.0, unitId));
  bytesStr = bytesStr.substr(0, bytesStr.find(".") + 3);
  return bytesStr + " " + units[unitId];
}

TEST(LoggingTest, LifeLikeLoad) {
  const size_t smallAllocs = 1000;
  const size_t bigAllocs = 10;
  // This test performs stress test to allocate and free a large of number of
  // array of variable sizes

  // std::stringstream log;
  // adapter_->setLogStream(&log);
  // adapter_->setLoggingEnabled(0x2);
  std::vector<af::array> allocs;
  for (int batcn = 0; batcn < 200; ++batcn) {
    for (int sml = 0; sml < smallAllocs; ++sml) {
      auto dimsArr = (af::randu(4, af::dtype::s32)) % 4 + 4;
      std::vector<int> dims(4);
      dimsArr.as(af::dtype::s32).host(dims.data());
      // std::cout << "small size="
      //           << formatMemory(dims[0] * dims[1] * dims[2] * dims[3] * 4)
      //           << std::endl;
      EXPECT_NO_THROW(
          allocs.push_back(af::array(dims[0], dims[1], dims[2], dims[3])));
    }
    for (int big = 0; big < bigAllocs; ++big) {
      auto dimsArr = (af::randu(4, af::dtype::s32)) % 100 + 100;
      std::vector<int> dims(4);
      dimsArr.as(af::dtype::s32).host(dims.data());
      // std::cout << "large size="
      //           << formatMemory(dims[0] * dims[1] * dims[2] * dims[3] * 4)
      //           << std::endl;
      EXPECT_NO_THROW(
          allocs.push_back(af::array(dims[0], dims[1], dims[2], dims[3])));
    }

    // Free all the big allocs and most of the other ones
    int freeNum = allocs.size() * 0.9;
    for (int freeMem = 0; freeMem < bigAllocs + smallAllocs * 0.8; ++freeMem) {
      allocs.pop_back();
    }
  }
}

// TEST_F(CachingMemoryManagerTest, OOM) {
//   af_backend b;
//   af_get_active_backend(&b);
//   // Despite that test is trying to allocate PB of memory,
//   // depending on the drivers, afopencl does not seem to guarantee to send an
//   // OOM signal. https://github.com/arrayfire/arrayfire/issues/2650 At the
//   // moment, skipping afopencl.
//   if (b == AF_BACKEND_OPENCL)
//     GTEST_SKIP();
//   af::array a;
//   // N^3 tensor means about 3PB: expected to OOM on today's cuda GPU.
//   const unsigned N = 99999;
//   try {
//     a = af::randu({N, N, N}, f32);
//   } catch (af::exception& ex) {
//     ASSERT_EQ(ex.err(), AF_ERR_NO_MEM);
//   } catch (...) {
//     EXPECT_TRUE(false) << "CachingMemoryManagerTest OOM: unexpected exception";
//   }
// }

// void testFragmentation(
//     std::shared_ptr<fl::MemoryManagerDeviceInterface> deviceInterface_,
//     std::shared_ptr<fl::CachingMemoryManager> adapter_,
//     bool expectOOM) {
//   af::Backend b = af::getActiveBackend();

//   if (b != AF_BACKEND_CUDA) {
//     GTEST_SKIP()
//         << "CachingMemoryManager fragmentation tests require CUDA backend";
//   }

//   const auto mms = deviceInterface_->getMaxMemorySize(0);
//   const auto maxNumf32 = mms / sizeof(float); // AF f32 is supposed to be 32b
//   ASSERT_NE(mms, 0);
//   {
//     af::array a1(.5f * maxNumf32);
//     adapter_->printInfo("After creating a1:", 0);
//   } // The a1 buffer will not be freed here, just registered to the cache
//   adapter_->printInfo("After releasing a1:", 0);

//   af::array a2(.1f * maxNumf32);
//   adapter_->printInfo("After creating a2:", 0);

//   af::array a3;
//   try {
//     a3 = af::array(.5f * maxNumf32);
//   } catch (af::exception& ex) {
//     if (expectOOM) {
//       ASSERT_EQ(ex.err(), AF_ERR_NO_MEM);
//     } else {
//       EXPECT_TRUE(false)
//           << "CachingMemoryManagerTest fragmentaiton not supposed to throw: "
//           << ex.what();
//     }
//   }
// }

// TEST_F(CachingMemoryManagerTest, Fragmentation) {
//   testFragmentation(deviceInterface_, adapter_, true); // should OOM
// }

// TEST_F(CachingMemoryManagerTest, RecLimit) {
//   constexpr static size_t ONE_GB = 1 << 30;
//   // Fine set the manager in order not to recycle big tensors:
//   adapter_->setRecyclingSizeLimit(2 * ONE_GB);
//   testFragmentation(deviceInterface_, adapter_, false); // should not OOM
// }

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  fl::init();
  return RUN_ALL_TESTS();
}
