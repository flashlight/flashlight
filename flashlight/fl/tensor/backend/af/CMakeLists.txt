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

# TODO{fl::Tensor} -- dynamically select which FL_ARRAYFIRE_USE options are
# correct based on ArrayFire backends which are located since the FL_BACKEND
# flag will go away soon. Implement this when implementing a
# backend registry CMake abstraction, i.e.
#
# fl_register_backend(
#   NAME ArrayFire
#   PLATFORM CUDA
#   TARGET ArrayFire::afcuda
# )
#
# fl_register_backend(
#   NAME ArrayFire
#   PLATFORM CPU
#   TARGET ArrayFire::afcpu
# )

# Check the proper ArrayFire backend is present
if (FL_USE_CPU AND NOT ArrayFire_CPU_FOUND)
  message(FATAL_ERROR "ArrayFire CPU not found: cannot build CPU backend")
elseif (FL_USE_CUDA AND NOT ArrayFire_CUDA_FOUND)
  message(FATAL_ERROR "ArrayFire CUDA not found: cannot build CUDA backend")
elseif (FL_USE_OPENCL AND NOT ArrayFire_OpenCL_FOUND)
  message(FATAL_ERROR "ArrayFire OpenCL not found: cannot build OpenCL backend")
endif()

# Set up variables to link the right ArrayFire backend
# TODO{fl::Tensor} enforce that we can only link AF if exactly one backend is
# found to avoid ambiguity (since we can only link one at a time)
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

target_compile_definitions(
  flashlight
  PUBLIC
  # AF backends are guaranteed to be found as specified
  # TODO{fl::Tensor} swap this to be dynamic based on whatever targets are found
  FL_ARRAYFIRE_USE_CPU=$<BOOL:${FL_USE_CPU}>
  FL_ARRAYFIRE_USE_CUDA=$<BOOL:${FL_USE_CUDA}>
  FL_ARRAYFIRE_USE_OPENCL=$<BOOL:${FL_USE_OPENCL}>
)

# ----------------------------- Sources -----------------------------
target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireTensor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  )

if (FL_USE_CUDA)
  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/AdvancedIndex.cu
  )
else()
  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/AdvancedIndex.cpp
    )
endif()
