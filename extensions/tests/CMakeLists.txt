cmake_minimum_required(VERSION 3.5.1)

set(FL_EXTENSION_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Download and build googletest
include(${FL_EXTENSION_TEST_DIR}/cmake/BuildGoogleTest.cmake)

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
    fl-extensions
    ${GTEST_LIBRARIES}
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${GTEST_INCLUDE_DIR}
    )
  target_compile_definitions(
    ${target}
    PRIVATE
    "ARCHDIR=\"${FL_EXTENSION_TEST_DIR}/common\""
    )
  add_test(${target} ${target})
endfunction(build_test)

build_test(${FL_EXTENSION_TEST_DIR}/common/SequentialBuilderTest.cpp)
