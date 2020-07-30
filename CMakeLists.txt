cmake_minimum_required(VERSION 3.5.1)

project(flashlight)

# ----------------------------- Configuration -----------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(FL_BUILD_TESTS "Build tests for wav2letter++" ON)
option(FL_BUILD_EXAMPLES "Build examples for wav2letter++" ON)
option(FL_BUILD_EXPERIMENTAL "Build internal experimental components for wav2letter++" OFF)
option(FL_BUILD_SCRIPTS "Build internal scripts for wav2letter++" OFF)
option(FL_BUILD_RECIPES "Build recipes" ON)
option(FL_BUILD_TOOLS "Build audio tools" OFF)

# --------------------------- Flashlight core ---------------------------

set(FLASHLIGHT_CORE_DIR "${CMAKE_SOURCE_DIR}/flashlight") # module root
add_subdirectory(${FLASHLIGHT_CORE_DIR})

set(FLASHLIGHT_LIB_DIR "${CMAKE_SOURCE_DIR}/libraries") # module root
include(${FLASHLIGHT_LIB_DIR}/CMakeLists.txt)

set(FLASHLIGHT_EXT_DIR "${CMAKE_SOURCE_DIR}/extensions") # module root
add_subdirectory(${FLASHLIGHT_EXT_DIR})

# --------------------------- Configure Tasks ---------------------------

set(FLASHLIGHT_TASKS_DIR "${CMAKE_SOURCE_DIR}/tasks") # module root
add_subdirectory(${FLASHLIGHT_TASKS_DIR})
