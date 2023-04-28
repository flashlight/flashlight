cmake_minimum_required(VERSION 3.16)

include(FetchContent)

set(gtest_URL https://github.com/google/googletest.git)
set(gtest_TAG release-1.12.1)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY ${gtest_URL}
  GIT_TAG        ${gtest_TAG}
)

# For Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(GTEST_LINKED_AS_SHARED_LIBRARY ${BUILD_SHARED_LIBS} CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)
