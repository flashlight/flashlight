cmake_minimum_required(VERSION 3.16)

option(FL_BUILD_CONTRIB "Build and link additional non-stabl contrib components" ON)

# Builds optional non-stable components of the FL API
if (FL_BUILD_CONTRIB)
  message(STATUS "Building Flashlight with contrib components.")
  include(${CMAKE_CURRENT_LIST_DIR}/contrib/CMakeLists.txt)
endif()

# ------------------------ Global External Dependencies ------------------------
# If cereal is found in a user-defined location, use it rather than
# downloading from source
find_package(cereal CONFIG)
if (NOT TARGET cereal::cereal)
  if (NOT FL_BUILD_STANDALONE)
    message(FATAL_ERROR "Required dependency cereal not found")
  endif()
  message(STATUS "Could not find cereal. Will download from source.")
  include(${PROJECT_SOURCE_DIR}/cmake/BuildCereal.cmake)
endif()
target_link_libraries(flashlight PUBLIC $<BUILD_INTERFACE:cereal::cereal>)

# -------------------------------- Components --------------------------------
# Tensor -- resolve backends first
include(${CMAKE_CURRENT_LIST_DIR}/tensor/CMakeLists.txt)

# Runtime
include(${CMAKE_CURRENT_LIST_DIR}/runtime/CMakeLists.txt)

# Common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

# Autograd
include(${CMAKE_CURRENT_LIST_DIR}/autograd/CMakeLists.txt)

# Dataset
include(${CMAKE_CURRENT_LIST_DIR}/dataset/CMakeLists.txt)

# Distributed
include(${CMAKE_CURRENT_LIST_DIR}/distributed/CMakeLists.txt)

# Meter
include(${CMAKE_CURRENT_LIST_DIR}/meter/CMakeLists.txt)

# NN
include(${CMAKE_CURRENT_LIST_DIR}/nn/CMakeLists.txt)

# Optim
include(${CMAKE_CURRENT_LIST_DIR}/optim/CMakeLists.txt)

# ----------------------- Examples and Tests ------------------------

if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test)
endif ()

if (FL_BUILD_EXAMPLES)
  set(FL_EXAMPLES_DIR ${CMAKE_CURRENT_LIST_DIR}/examples)
  add_subdirectory(${FL_EXAMPLES_DIR})
  install(
    DIRECTORY ${FL_EXAMPLES_DIR}
    DESTINATION ${FL_INSTALL_EXAMPLES_DIR}
    COMPONENT examples
  )
endif()
