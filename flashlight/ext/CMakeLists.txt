cmake_minimum_required(VERSION 3.13)

# ------------------------- Components -------------------------

include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

cmake_dependent_option(FL_EXT_BUILD_INTEGRATIONS
  "Build and link Halide integration" ON
  "FL_BUILD_CORE" OFF)

if (FL_EXT_BUILD_INTEGRATIONS)
  include(${CMAKE_CURRENT_LIST_DIR}/integrations/CMakeLists.txt)
endif()

# --------------------------- Tests ---------------------------

if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()
