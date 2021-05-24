cmake_minimum_required(VERSION 3.10)

# ------------------------- Components -------------------------

include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

# --------------------------- Tests ---------------------------

if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()
