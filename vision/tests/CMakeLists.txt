cmake_minimum_required(VERSION 3.5.1)

# Download and build googletest
#include(${CMAKE_MODULE_PATH}/BuildGoogleTest.cmake)
#if (
message(STATUS "Building tests HERE " ${GTEST_LIBRARIES})

function(build_test SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  add_dependencies(${target} gtest) # make sure gtest is built first
  target_link_libraries(
    ${target}
    PRIVATE
    flashlight
    flashlightVision
    ${GTEST_LIBRARIES}
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${GTEST_INCLUDE_DIR}
    )
  add_test(${target} ${target})
endfunction(build_test)

set(FLASHLIGHT_TEST_DIR ${CMAKE_SOURCE_DIR}/vision/tests)
if (FL_BUILD_TESTS)
  build_test(${FLASHLIGHT_TEST_DIR}/criterion/SetCriterionTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/TransformerTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/PositionalEmbeddingSineTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/criterion/HungarianTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/criterion/HungarianMatcherTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/dataset/BoxUtilsTest.cpp)
endif ()
