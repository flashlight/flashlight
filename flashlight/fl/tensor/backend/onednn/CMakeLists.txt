cmake_minimum_required(VERSION 3.16)

# ----------------------------- OneDNN -----------------------------

# Try to find oneDNN
find_package(DNNL CONFIG)
if (NOT DNNL_FOUND AND FL_USE_ONEDNN)
  message(FATAL_ERROR
    "oneDNN required to build oneDNN backend: oneDNN was not found")
else()
  message(STATUS "oneDNN found")
endif()

# Need to link against OpenMP if DNNL was built with the OpenMP CPU backend
if (DNNL_CPU_RUNTIME STREQUAL "OMP" OR DNNL_CPU_THREADING_RUNTIME STREQUAL "OMP")
  find_package(OpenMP REQUIRED)
  target_link_libraries(flashlight PRIVATE OpenMP::OpenMP_CXX)
endif()

# TODO: conditionally set this to use OPENCL if the detected
# DNNL is built with OpenCL support
fl_set_backend_state(ENABLE CPU) # enables FL_USE_CPU

# ----------------------------- Sources -----------------------------
find_package(MKL)
if (NOT MKL_FOUND)
  message(STATUS "Using stdlib random number generation")
  set(${FL_USE_MKL_RNG} FALSE)
else()
  message(STATUS "MKL found - will use for random number generation")
  set(${FL_USE_MKL_RNG} TRUE)
endif()

target_compile_definitions(
  flashlight
  PRIVATE
  FL_USE_MKL_RNG=$<BOOL:${FL_USE_MKL_RNG}>
)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnCPUStream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/OneDnnTensor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
)

target_include_directories(
  flashlight
  PUBLIC
  ${MKL_INCLUDE_DIR}
)

target_link_libraries(
  flashlight
  PUBLIC
  DNNL::dnnl
  ${MKL_LIBRARIES}
)
