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
    experimental_memory
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

if (FL_BUILD_TESTS)
  build_test(${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/tests/FreeListTest.cpp)
  build_test(${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/tests/MemoryPoolTest.cpp)
  build_test(${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/tests/CompositeMemoryAllocatorTest.cpp)
  build_test(${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/tests/ConfigurableMemoryAllocatorTest.cpp)
endif ()
