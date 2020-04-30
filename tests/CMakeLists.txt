cmake_minimum_required(VERSION 3.5.1)

find_package(GTest 1.10.0)

if (NOT TARGET gtest AND NOT GTEST_FOUND)
  message(STATUS "googletest not found - will download and build from source")
  # Download and build googletest
  include(${CMAKE_MODULE_PATH}/BuildGoogleTest.cmake)
else()
  message(STATUS "gtest found: (include: ${GTEST_INCLUDE_DIRS}, lib: ${GTEST_BOTH_LIBRARIES}")
endif()

function(build_test SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  if (TARGET gtest)
    add_dependencies(${target} gtest) # make sure gtest is built first
  endif()
  target_link_libraries(
    ${target}
    PRIVATE
    flashlight
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

set(FLASHLIGHT_TEST_DIR ${CMAKE_SOURCE_DIR}/tests)
if (FL_BUILD_TESTS)
  build_test(${FLASHLIGHT_TEST_DIR}/autograd/AutogradTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/common/DevicePtrTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/common/HistogramTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/common/LoggingTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/common/SerializationTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/optim/OptimTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/memory/FreeListTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/memory/MemoryPoolTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/memory/CachingMemoryManagerTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/memory/CompositeMemoryAllocatorTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/memory/ConfigurableMemoryAllocatorTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/memory/MemoryFrameworkTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/nn/ModuleTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/nn/NNSerializationTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/nn/NNUtilsTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/dataset/DatasetTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/dataset/DatasetUtilsTest.cpp)
  build_test(${FLASHLIGHT_TEST_DIR}/meter/MeterTest.cpp)
  if (FL_BUILD_DISTRIBUTED)
    build_test(${FLASHLIGHT_TEST_DIR}/distributed/AllReduceTest.cpp)
  endif ()
  if (FL_BUILD_CONTRIB)
    build_test(${FLASHLIGHT_TEST_DIR}/contrib/modules/ContribModuleTest.cpp)
    build_test(${FLASHLIGHT_TEST_DIR}/contrib/modules/ContribSerializationTest.cpp)
  endif ()
endif ()
