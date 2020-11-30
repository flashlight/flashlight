# CUDAUtils - general utilities for working with CUDA
#
# This file should only be included if CUDA is to be linked (e.g. CUDA is the specified
# criterion backend), as it searches for CUDA.
#

### Find CUDA
find_package(CUDA 9.2 QUIET) # CUDA >= 9.2 is required for >= ArrayFire 3.6.1
if (CUDA_FOUND)
  message(STATUS "CUDA found (library: ${CUDA_LIBRARIES} include: ${CUDA_INCLUDE_DIRS})")
else()
  message(FATAL_ERROR "CUDA required to build CUDA backend")
endif()

# This line must be placed after find_package(CUDA)
include(${CMAKE_MODULE_PATH}/select_compute_arch.cmake)

### Set compilation flags
# NVCC doesn't properly listen to cxx version flags, so manually override.
# This MUST be done after CUDA is found, but before any cuda libs/binaries have
# been created.
function (set_cuda_cxx_compile_flags)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++14" PARENT_SCOPE)
  # Using host flags makes things bad - keep things clean
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endfunction()

### Detect GPU architectures
# Detect architectures (see select_compute_arch.cmake) and
# add appropriate flags to nvcc for gencode/arch/ptx
function (set_cuda_arch_nvcc_flags)
  set(
    CUDA_architecture_build_targets
    "Common"
    CACHE STRING "Detected CUDA architectures for this build"
    )
  cuda_select_nvcc_arch_flags(cuda_arch_flags ${CUDA_architecture_build_targets})
  message(STATUS "CUDA architecture flags: " ${cuda_arch_flags})
  # Add to flag list
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${cuda_arch_flags}" PARENT_SCOPE)
  mark_as_advanced(CUDA_architecture_build_targets)
endfunction()

function (cuda_enable_position_independent_code)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC" PARENT_SCOPE)
endfunction ()
