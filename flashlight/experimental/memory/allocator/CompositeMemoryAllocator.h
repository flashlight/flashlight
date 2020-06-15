/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashlight/experimental/memory/allocator/MemoryAllocator.h"

namespace fl {

// CompositeMemoryAllocator reduces external fragmentation by grouping similar
// size objects to the same allocator. It is a collection of simpler memory
// allocators ordered by maximum allocation size, from small to large.
// Allocation requests are delegated to the allocatos whos maximum allocation
// size is the nearest larger value. If that allocator fails to satisfy the
// allocation, then the request is delegated to the next one up, and so on.
class CompositeMemoryAllocator : public MemoryAllocator {
 public:
  explicit CompositeMemoryAllocator(std::string name);
  ~CompositeMemoryAllocator() override = default;

  struct AllocatorAndCriteria {
    size_t maxAllocationSize;
    std::unique_ptr<MemoryAllocator> allocator;

    // Support for std::sort() of AllocatorAndCriteria collection.
    bool operator<(const AllocatorAndCriteria& other) const {
      return (maxAllocationSize < other.maxAllocationSize);
    }

    // CompositeMemoryAllocator{stats=Stats{
    // highlights={size=19999975712(18GB+641MB+474KB+288) oomEventCount=86695
    // maxInternalFragmentationScore=0.270346 currentlyAllocatedCnt=2499}
    // arena=0x1000 blockSize=32606(31KB+862) allocationsCount=99999
    // deAllocationsCount=97500 internalFragmentationScore=0.201048
    // externalFragmentationScore=0.0418752
    // maxInternalFragmentationScore=0.270346
    // maxExternalFragmentationScore=0.0423482 oomEventCount=86695
    // performanceCost=207031944(20m+703k+1944)
    //  statsInBytes={arenaSize=19999975712(18GB+641MB+474KB+288)
    //  freeCount=19830702617(18GB+480MB+32KB+537)
    //  allocatedCount=169273095(161MB+441KB+775) allocatedRatio=0.00846367
    //  maxAllocatedCount=834278956(795MB+645KB+556) maxAllocatedRatio=0.041714}
    // } statsInBlocks={arenaSize=808852(789KB+916) freeCount=763178(745KB+298)
    // allocatedCount=45674(44KB+618) allocatedRatio=0.0564677
    // maxAllocatedCount=223503(218KB+271) maxAllocatedRatio=0.276321}}
    // subArenaStats={
    // Stats{
    // highlights={size=199968(195KB+288) maxInternalFragmentationScore=0.96875
    // currentlyAllocatedCnt=1} arena=0x4a8146c00 blockSize=32
    // allocationsCount=55 deAllocationsCount=54
    // internalFragmentationScore=0.46875 externalFragmentationScore=0
    // maxInternalFragmentationScore=0.96875 maxExternalFragmentationScore=0
    // oomEventCount=0 performanceCost=109
    //  statsInBytes={arenaSize=199968(195KB+288) freeCount=199951(195KB+271)
    //  allocatedCount=17 allocatedRatio=8.50136e-05 maxAllocatedCount=160
    //  maxAllocatedRatio=0.000800128}
    // } statsInBlocks={arenaSize=6249 freeCount=6248 allocatedCount=1
    // allocatedRatio=0.000160026 maxAllocatedCount=9
    // maxAllocatedRatio=0.00144023}}
    // }
    // Stats{
    // highlights={size=99998720(95MB+375KB) oomEventCount=86695
    // maxAllocatedRatio=1 maxExternalFragmentationScore=0.988384
    // currentlyAllocatedCnt=332} arena=0x4a21e9000 blockSize=512
    // allocationsCount=13249 deAllocationsCount=12917
    // internalFragmentationScore=0.00396341 externalFragmentationScore=0.981749
    // maxInternalFragmentationScore=0.00620026
    // maxExternalFragmentationScore=0.988384 oomEventCount=86695
    // performanceCost=5923916(592k+3916)
    //  statsInBytes={arenaSize=99998720(95MB+375KB)
    //  freeCount=79610600(75MB+944KB+744)
    //  allocatedCount=20388120(19MB+454KB+280) allocatedRatio=0.203884
    //  maxAllocatedCount=99642877(95MB+27KB+509) maxAllocatedRatio=0.996442}
    // } statsInBlocks={arenaSize=195310(190KB+750) freeCount=155331(151KB+707)
    // allocatedCount=39979(39KB+43) allocatedRatio=0.204695
    // maxAllocatedCount=195310(190KB+750) maxAllocatedRatio=1}}
    // }
    // Stats{
    // highlights={size=19899777024(18GB+545MB+928KB)
    // maxInternalFragmentationScore=0.271667 currentlyAllocatedCnt=2166}
    // arena=0x1000 blockSize=32768(32KB) allocationsCount=86695
    // deAllocationsCount=84529 internalFragmentationScore=0.202036
    // externalFragmentationScore=0.0371527
    // maxInternalFragmentationScore=0.271667
    // maxExternalFragmentationScore=0.0375947 oomEventCount=0
    // performanceCost=201107919(20m+110k+7919)
    //  statsInBytes={arenaSize=19899777024(18GB+545MB+928KB)
    //  freeCount=19750892066(18GB+403MB+940KB+546)
    //  allocatedCount=148884958(141MB+1011KB+478) allocatedRatio=0.00748174
    //  maxAllocatedCount=734635919(700MB+617KB+911)
    //  maxAllocatedRatio=0.0369168}
    // } statsInBlocks={arenaSize=607293(593KB+61) freeCount=601599(587KB+511)
    // allocatedCount=5694 allocatedRatio=0.00937603
    // maxAllocatedCount=28184(27KB+536) maxAllocatedRatio=0.0464092}}
    // }
    // }}
    // CompositeMemoryAllocator currenly allocated higtogram:
    // HistogramStats{ min=[    0] max_=[ 633K] sum=[2375M] mean=[  32K]
    //                  numValues=[75130] numBuckets=[15]
    // [    0-  42K] 48102: **************************************************
    // [  42K-  84K] 12981: *************
    // [  84K- 126K]  6720: *******
    // [ 126K- 168K]  3428: ****
    // [ 168K- 211K]  1837: **
    // [ 211K- 253K]   983: *
    // [ 253K- 295K]   520: *
    // [ 295K- 337K]   259:
    // [ 337K- 380K]   139:
    // [ 380K- 422K]    74:
    // [ 422K- 464K]    41:
    // [ 464K- 506K]    27:
    // [ 506K- 548K]    14:
    // [ 548K- 591K]     2:
    // [ 591K- 633K]     3:
    // }
    std::string prettyString() const;
  };

  struct FirstOomStats {
    FirstOomStats();
    std::string prettyString() const;

    Stats stats;
    size_t allocationSize;
    std::string description;
  };

  void* allocate(size_t size) override;
  void free(void* ptr) override;

  // Retunes weighted (by relative memory size) summary of internal allocators
  // stats.
  Stats getStats() const override;
  size_t getAllocatedSizeInBytes(void* ptr) const override;
  void setLogLevel(int logLevel) override;
  bool jitTreeExceedsMemoryPressure(size_t bytes) override;

  std::string prettyString() const override;

  void add(AllocatorAndCriteria allocatorAndCriteria);

 private:
  // Allocation used for keeping track of allocated memory objects. It is
  // used by free() to know to which allocator we want to return that memory
  // and for stats calculations.
  struct Allocation {
    size_t size;
    size_t allocatorsAndCriteriasIndex;
  };

  std::vector<AllocatorAndCriteria> allocatorsAndCriterias_;
  std::unordered_map<void*, Allocation> ptrToAllocation_;
  size_t totalNumberOfAllocations_;
  size_t arenaSizeInBlocks_;
  size_t arenaSizeInBytes_;
  bool failToAllocate_;
  size_t oomEventCount_;
  FirstOomStats firstOomStats_;
};

} // namespace fl
