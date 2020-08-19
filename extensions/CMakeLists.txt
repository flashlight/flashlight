cmake_minimum_required(VERSION 3.5.1)

# ------------------------- Components -------------------------

# common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)
 
# --------------------------- Tests ---------------------------

# Build tests
option(FL_BUILD_TESTS "Build tests for flashlight" ON)
if (FL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
endif ()
