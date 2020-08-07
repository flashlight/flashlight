cmake_minimum_required(VERSION 3.5.1)

# ------------------------- Library -------------------------
# TODO(jacobkahn): Task libraries aren't currently installed.
# Having them bundled into flashlight and moving headers accordingly is
# one potential path forward here.
add_library(
  flashlight-task-speech-recognition
  ""
  )

# ------------------------- Dependency Fallback -------------------------

if (FL_LIBRARIES_USE_MKL)
  find_package(MKL)
  if (NOT MKL_FOUND)
    message(WARNING "MKL not found; forcing FL_LIBRARIES_USE_MKL=OFF.")
    set(FL_LIBRARIES_USE_MKL OFF CACHE BOOL "" FORCE)
  endif ()
endif ()

# ------------------------ Global External Dependencies ------------------------
# TODO(jacobkahn): add gating for Conan glog and gflags packages

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

target_link_libraries(
  flashlight-task-speech-recognition
  PUBLIC
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${MKL_LIBRARIES}
  flashlight
  )

target_include_directories(
  flashlight-task-speech-recognition
  PUBLIC
  ${GFLAGS_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${MKL_INCLUDE_DIR}
  )

# ------------------------ Components ------------------------

# Common
include(${CMAKE_CURRENT_LIST_DIR}/common/CMakeLists.txt)

# Criterion
include(${CMAKE_CURRENT_LIST_DIR}/criterion/CMakeLists.txt)

# Data
include(${CMAKE_CURRENT_LIST_DIR}/data/CMakeLists.txt)

# Decoder
include(${CMAKE_CURRENT_LIST_DIR}/decoder/CMakeLists.txt)

# Runtime
include(${CMAKE_CURRENT_LIST_DIR}/runtime/CMakeLists.txt)

# ----------------------------- Binaries -----------------------------
add_executable(Train ${CMAKE_CURRENT_LIST_DIR}/Train.cpp)
add_executable(Test ${CMAKE_CURRENT_LIST_DIR}/Test.cpp)
add_executable(Decoder ${CMAKE_CURRENT_LIST_DIR}/Decode.cpp)

target_link_libraries(Train flashlight-task-speech-recognition)
target_link_libraries(Test flashlight-task-speech-recognition)
target_link_libraries(Decoder flashlight-task-speech-recognition)

# --------------------------- Tests ---------------------------

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests)
endif ()
