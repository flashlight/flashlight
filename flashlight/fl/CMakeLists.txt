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
find_package(cereal)
if (NOT TARGET cereal AND NOT cereal_FOUND AND FL_BUILD_STANDALONE)
  message(STATUS "cereal NOT found. Will download from source")
  set(CEREAL_INSTALL_PATH ${FL_INSTALL_INC_DIR}/cereal)
  include(${CMAKE_MODULE_PATH}/BuildCereal.cmake)
  add_dependencies(flashlight cereal)
  # Move cereal headers at install time
  install(DIRECTORY ${CEREAL_SOURCE_DIR}/include/cereal/
    DESTINATION ${CEREAL_INSTALL_PATH}
    COMPONENT cereal
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN ".git" EXCLUDE
    )
  install(FILES ${CEREAL_SOURCE_DIR}/LICENSE ${CEREAL_SOURCE_DIR}/README.md
    DESTINATION ${CEREAL_INSTALL_PATH}
    )
  target_include_directories(flashlight PUBLIC ${cereal_INCLUDE_DIRS})
else()
  message(STATUS "Found cereal")
  target_link_libraries(flashlight PRIVATE cereal)
endif()
setup_install_find_module(${CMAKE_MODULE_PATH}/Findcereal.cmake)

# -------------------------------- Components --------------------------------
# Tensor -- resolve backends and compute runtimes first
include(${CMAKE_CURRENT_LIST_DIR}/tensor/CMakeLists.txt)

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

# Runtime
include(${CMAKE_CURRENT_LIST_DIR}/runtime/CMakeLists.txt)

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
