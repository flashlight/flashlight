cmake_minimum_required(VERSION 3.16)

include(FetchContent)

set(gtest_URL https://github.com/google/googletest.git)
set(gtest_TAG release-1.12.1)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY ${gtest_URL}
  GIT_TAG        ${gtest_TAG}
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE) # for Windows
FetchContent_MakeAvailable(googletest)
