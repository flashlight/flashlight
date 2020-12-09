cmake_minimum_required(VERSION 3.5.1)

# Get or find Google Test
find_package(GTest 1.10.0)
if (NOT TARGET gtest AND NOT GTEST_FOUND)
  message(STATUS "googletest not found - will download and build from source")
  # Download and build googletest
  include(${CMAKE_MODULE_PATH}/BuildGoogleTest.cmake) # internally sets GTEST_LIBRARIES
else()
  message(STATUS "gtest found: (include: ${GTEST_INCLUDE_DIRS}, lib: ${GTEST_BOTH_LIBRARIES}")
  # Try again with a config to make sure there isn't some broken module in the way
  find_package(GTest CONFIG 1.10.0)
  if (NOT TARGET GTest::gtest)
    # If we found the weirdly-named CMake targets from FindGTest, alias them.
    # Assume gmock was built into gtest as it should be with >= 1.10.0:
    add_library(GTest::gtest ALIAS GTest::GTest)
    add_library(GTest::gtest_main ALIAS GTest::Main)
    add_library(GTest::gmock ALIAS GTest::GTest)
    add_library(GTest::gmock_main ALIAS GTest::Main)
  endif()
endif()

include(GoogleTest)

function(build_test SRCFILE LINK_LIBRARY PREPROC_DEFS)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  if (TARGET gtest)
    add_dependencies(${target} gtest) # make sure gtest is built first
  endif()
  target_link_libraries(
    ${target}
    PUBLIC
    ${LINK_LIBRARY}
    GTest::gtest
    GTest::gtest_main
    GTest::gmock
    GTest::gmock_main
    )
  target_include_directories(
    ${target}
    PUBLIC
    ${PROJECT_SOURCE_DIR}
    )
  target_compile_definitions(
    ${target}
    PUBLIC
    ${PREPROC_DEFS}
    )
  gtest_discover_tests(${target})
endfunction(build_test)
