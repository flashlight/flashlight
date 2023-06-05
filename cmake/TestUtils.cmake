cmake_minimum_required(VERSION 3.5.1)

set(GTEST_IMPORTED_TARGETS "")

# Get or find Google Test and Google Mock
find_package(GTest 1.10.0)
if (NOT GTEST_FOUND)
  if (NOT TARGET gtest)
    message(STATUS "googletest not found - will download and build from source")
    # Download and build googletest
    include(${PROJECT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)
    list(APPEND GTEST_IMPORTED_TARGETS GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main)
  endif()
else()
  message(STATUS "gtest found: (include: ${GTEST_INCLUDE_DIRS}, lib: ${GTEST_BOTH_LIBRARIES}")
  if (TARGET GTest::GTest)
    # We found the differently-named CMake targets from FindGTest
    if (NOT TARGET GTest::Main)
      message(FATAL_ERROR "Google Test must be built with main")
    endif()
    list(APPEND GTEST_IMPORTED_TARGETS GTest::GTest GTest::Main)
  endif()
  if (NOT TARGET GTest::gmock)
    find_package(GMock REQUIRED)
    message(STATUS "gmock found: (include: ${GMOCK_INCLUDE_DIRS}, lib: ${GMOCK_BOTH_LIBRARIES})")
  endif()
  list(APPEND GTEST_IMPORTED_TARGETS GTest::gmock GTest::gmock_main)
  message(STATUS "Found gtest and gmock on system.")
endif()

include(GoogleTest)
find_package(Threads REQUIRED)

function(build_test)
  set(options)
  set(oneValueArgs SRC)
  set(multiValueArgs LIBS PREPROC)
  cmake_parse_arguments(build_test "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN})

  get_filename_component(src_name ${build_test_SRC} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${build_test_SRC})
  if (TARGET gtest)
    add_dependencies(${target} gtest) # make sure gtest is built first
  endif()
  target_link_libraries(
    ${target}
    PUBLIC
    ${GTEST_IMPORTED_TARGETS}
    ${build_test_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    )
  target_include_directories(
    ${target}
    PUBLIC
    ${PROJECT_SOURCE_DIR}
    )
  target_compile_definitions(
    ${target}
    PUBLIC
    ${build_test_PREPROC}
    )
  if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_definitions(
      ${target}
      PUBLIC
      GTEST_LINKED_AS_SHARED_LIBRARY=$<BOOL:${BUILD_SHARED_LIBS}>
      GMOCK_LINKED_AS_SHARED_LIBRARY=$<BOOL:${BUILD_SHARED_LIBS}>
    )
  endif()
  gtest_add_tests(TARGET ${target})
endfunction(build_test)
