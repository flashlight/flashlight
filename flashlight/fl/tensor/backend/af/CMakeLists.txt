cmake_minimum_required(VERSION 3.10)

# ----------------------------- ArrayFire -----------------------------
find_package(ArrayFire REQUIRED)
if (ArrayFire_FOUND AND ArrayFire_VERSION VERSION_LESS 3.7.1)
  message(FATAL_ERROR "ArrayFire versions < 3.7.1 are no longer supported "
    "with flashlight. To build flashlight with a version of ArrayFire "
    "< 3.7.1, use commit <= 5518d91b7f4fd5b400cbc802cfbecc0df57836bd.")
endif()

if (ArrayFire_FOUND)
  message(STATUS "ArrayFire found (include: ${ArrayFire_INCLUDE_DIRS}, library: ${ArrayFire_LIBRARIES})")
else()
  message(FATAL_ERROR "ArrayFire not found")
endif()

# Check the proper ArrayFire backend is present
if (FL_USE_CPU AND NOT ArrayFire_CPU_FOUND)
  message(FATAL_ERROR "ArrayFire CPU not found: cannot build CPU backend")
elseif (FL_USE_CUDA AND NOT ArrayFire_CUDA_FOUND)
  message(FATAL_ERROR "ArrayFire CUDA not found: cannot build CUDA backend")
elseif (FL_USE_OPENCL AND NOT ArrayFire_OpenCL_FOUND)
  message(FATAL_ERROR "ArrayFire OpenCL not found: cannot build OpenCL backend")
endif()

# Set up variables to link the right ArrayFire backend
if (FL_USE_CPU)
  set(FL_AF_BACKEND "afcpu")
elseif (FL_USE_CUDA)
  set(FL_AF_BACKEND "afcuda")
elseif (FL_USE_OPENCL)
  set(FL_AF_BACKEND "afopencl")
else()
  message(FATAL_ERROR "flashlight backend ill-specified")
endif()

# Link ArrayFire to flashlight
target_link_libraries(flashlight PUBLIC ArrayFire::${FL_AF_BACKEND})

# ----------------------------- Sources -----------------------------
target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireTensor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )
