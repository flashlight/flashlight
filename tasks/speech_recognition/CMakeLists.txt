cmake_minimum_required(VERSION 3.5.1)

project(flashlight_asr_task)

# ----------------------------- Configuration -----------------------------

option(W2L_LIBRARIES_USE_CUDA "Use CUDA in libraries-only build" ON)
option(W2L_LIBRARIES_USE_KENLM "Use KenLM in libraries-only build" ON)
option(W2L_LIBRARIES_USE_MKL "Use MKL in libraries-only build" ON)
option(W2L_BUILD_FOR_PYTHON "Build Python bindings" OFF)
option(W2L_BUILD_TESTS "Build tests for wav2letter++" ON)
option(W2L_BUILD_EXAMPLES "Build examples for wav2letter++" ON)
option(W2L_BUILD_EXPERIMENTAL "Build internal experimental components for wav2letter++" OFF)
option(W2L_BUILD_SCRIPTS "Build internal scripts for wav2letter++" OFF)
option(W2L_BUILD_RECIPES "Build recipes" ON)
option(W2L_BUILD_TOOLS "Build audio tools" OFF)
set(KENLM_MAX_ORDER 6 CACHE STRING "Maximum ngram order for KenLM")

# ------------------------- Dependency Fallback -------------------------

if (W2L_LIBRARIES_USE_MKL)
  find_package(MKL)
  if (NOT MKL_FOUND)
    message(WARNING "MKL not found; forcing W2L_LIBRARIES_USE_MKL=OFF.")
    set(W2L_LIBRARIES_USE_MKL OFF CACHE BOOL "" FORCE)
  endif ()
endif ()

# ------------------------ Global External Dependencies ------------------------
# Find GLog
find_package(GLOG REQUIRED)
if (GLOG_FOUND)
  message(STATUS "GLOG found")
else()
  message(FATAL_ERROR "GLOG not found")
endif()

# Find GFlags
find_package(GFLAGS REQUIRED)
if (GFLAGS_FOUND)
  message(STATUS "GFLAGS found")
else()
  message(FATAL_ERROR "GFLAGS not found")
endif()

# Find and setup OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  message(STATUS "OpenMP found")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}"
  )
else()
  message(STATUS "OpenMP not found - building without OpenMP")
endif()

# Find Flashlight
find_library(flashlight REQUIRED HINTS "${CMAKE_BINARY_DIR}/flashlight")
find_library(fl-extensions REQUIRED HINTS "${CMAKE_BINARY_DIR}/extensions")
find_library(fl-libraries REQUIRED HINTS "${CMAKE_BINARY_DIR}/libraries")

# ------------------------ Components ------------------------

# Common
add_subdirectory(${PROJECT_SOURCE_DIR}/common)

# Criterion
add_subdirectory(${PROJECT_SOURCE_DIR}/criterion)

# Data
add_subdirectory(${PROJECT_SOURCE_DIR}/data)

# Decoder
add_subdirectory(${PROJECT_SOURCE_DIR}/decoder)

# Runtime
add_subdirectory(${PROJECT_SOURCE_DIR}/runtime)

# ----------------------------- wav2letter++ lib  -----------------------------

add_library(
  wav2letter++
  ""
  )

set_target_properties(
  wav2letter++
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

target_link_libraries(
  wav2letter++
  PUBLIC
  common
  criterion
  data
  decoder
  runtime
  flashlight
  fl-extensions
  fl-libraries
  )

target_include_directories(
  wav2letter++
  PUBLIC
  ${PROJECT_SOURCE_DIR} # all includes are based at 
  )

# ----------------------------- Train -----------------------------
add_executable(
  Train
  Train.cpp
)

target_link_libraries(
  Train
  wav2letter++
  )

# ----------------------------- Test -----------------------------
add_executable(
  Test
  Test.cpp
  )

target_link_libraries(
  Test
  wav2letter++
  )

# ----------------------------- Decoder -----------------------------
add_executable(
  Decoder
  Decode.cpp
  )

target_link_libraries(
  Decoder
  wav2letter++
  )

# --------------------------- Tests ---------------------------

# Build tests
option(FL_BUILD_TESTS "Build tests for flashlight" ON)
if (FL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif ()