/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <future>
#include <string>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "libraries/common/System.h"

using namespace w2l;

TEST(SystemTest, PathsConcat) {
  auto path1 = pathsConcat("/tmp/", "test.wav");
  auto path2 = pathsConcat("/tmp", "test.wav");
  ASSERT_EQ(path1, "/tmp/test.wav");
  ASSERT_EQ(path2, "/tmp/test.wav");
}

static std::function<int(void)> makeSucceedsAfterIters(int iters) {
  auto state = std::make_shared<int>(0);
  return [state, iters]() {
    if (++*state >= iters) {
      return 42;
    } else {
      throw std::runtime_error("bleh");
    }
  };
}

static std::function<int(void)> makeSucceedsAfterMs(double ms) {
  using namespace std::chrono;
  auto state = std::make_shared<time_point<steady_clock>>();
  return [state, ms]() {
    auto now = steady_clock::now();
    if (state->time_since_epoch().count() == 0) {
      *state = now;
    }
    if (now - *state >= duration<double, std::milli>(ms)) {
      return 42;
    } else {
      throw std::runtime_error("bleh");
    }
  };
}

template <class Fn>
std::future<typename std::result_of<Fn()>::type> retryAsync(
    std::chrono::duration<double> initial,
    double factor,
    int64_t iters,
    Fn f) {
  return std::async(std::launch::async, [=]() {
    return retryWithBackoff(initial, factor, iters, f);
  });
}

TEST(SystemTest, RetryWithBackoff) {
  auto alwaysSucceeds = []() { return 42; };
  auto alwaysFails = []() -> int { throw std::runtime_error("bleh"); };

  std::vector<std::future<int>> goods;
  std::vector<std::future<int>> bads;
  std::vector<std::future<int>> invalids;

  auto ms0 = std::chrono::milliseconds(0);
  auto ms50 = std::chrono::milliseconds(50);

  goods.push_back(retryAsync(ms0, 1.0, 5, alwaysSucceeds));
  goods.push_back(retryAsync(ms50, 2.0, 5, alwaysSucceeds));

  bads.push_back(retryAsync(ms0, 1.0, 5, alwaysFails));
  bads.push_back(retryAsync(ms50, 2.0, 5, alwaysFails));

  bads.push_back(retryAsync(ms0, 1.0, 5, makeSucceedsAfterIters(6)));
  bads.push_back(retryAsync(ms50, 2.0, 5, makeSucceedsAfterIters(6)));
  goods.push_back(retryAsync(ms0, 1.0, 5, makeSucceedsAfterIters(5)));
  goods.push_back(retryAsync(ms50, 2.0, 5, makeSucceedsAfterIters(5)));

  bads.push_back(retryAsync(ms0, 1.0, 5, makeSucceedsAfterMs(999)));
  bads.push_back(retryAsync(ms50, 2.0, 5, makeSucceedsAfterMs(999)));
  bads.push_back(retryAsync(ms0, 1.0, 5, makeSucceedsAfterMs(500)));
  goods.push_back(retryAsync(ms50, 2.0, 5, makeSucceedsAfterMs(500)));

  invalids.push_back(retryAsync(-ms50, 2.0, 5, alwaysSucceeds));
  invalids.push_back(retryAsync(ms50, -1.0, 5, alwaysSucceeds));
  invalids.push_back(retryAsync(ms50, 2.0, 0, alwaysSucceeds));
  invalids.push_back(retryAsync(ms50, 2.0, -1, alwaysSucceeds));

  for (auto& fut : goods) {
    ASSERT_EQ(fut.get(), 42);
  }
  for (auto& fut : bads) {
    ASSERT_THROW(fut.get(), std::runtime_error);
  }
  for (auto& fut : invalids) {
    ASSERT_THROW(fut.get(), std::invalid_argument);
  }

  // check special case promise<void> / future<void>
  auto alwaysSucceedsVoid = []() -> void {};
  auto alwaysFailsVoid = []() -> void { throw std::runtime_error("bleh"); };

  retryAsync(ms0, 1.0, 5, alwaysSucceedsVoid).get();
  ASSERT_THROW(
      retryAsync(ms0, 1.0, 5, alwaysFailsVoid).get(), std::runtime_error);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Resolve directory for sample dictionary
#ifdef W2L_DICTIONARY_TEST_DIR
  loadPath = W2L_DICTIONARY_TEST_DIR;
#endif

  return RUN_ALL_TESTS();
}