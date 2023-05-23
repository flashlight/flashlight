cmake_minimum_required(VERSION 3.16)

# ----------------------------- ArrayFire -----------------------------
find_package(ArrayFire 3.7.3 REQUIRED)
if (ArrayFire_FOUND AND ArrayFire_VERSION VERSION_LESS 3.7.1)
  message(FATAL_ERROR "ArrayFire versions < 3.8.1 are no longer supported "
    "with flashlight. To build flashlight with a version of ArrayFire "
    "< 3.7.1, use commit <= 5518d91b7f4fd5b400cbc802cfbecc0df57836bd.")
endif()

if (ArrayFire_FOUND)
  message(STATUS "ArrayFire found (include: ${ArrayFire_INCLUDE_DIRS}, library: ${ArrayFire_LIBRARIES})")
else()
  message(FATAL_ERROR "ArrayFire not found")
endif()

# Backend-specific options for ArrayFire. Automatically set if only one ArrayFire backend
# target is found; if multiple are found, these must be manually specified.
option(FL_ARRAYFIRE_USE_CUDA "Use the ArrayFire CUDA backend" OFF)
option(FL_ARRAYFIRE_USE_CPU "Use the ArrayFire CPU backend" OFF)
option(FL_ARRAYFIRE_USE_OPENCL "Use the ArrayFire OpenCL backend" OFF)

macro(fl_check_af_backend_option OPTION TARGET)
  if (${OPTION} AND NOT TARGET ${TARGET})
    message(FATAL_ERROR "${OPTION} was set to ON but the ${TARGET} target was not found")
  endif()
endmacro()

fl_check_af_backend_option(FL_ARRAYFIRE_USE_CUDA ArrayFire::afcuda)
fl_check_af_backend_option(FL_ARRAYFIRE_USE_OPENCL ArrayFire::afopencl)
fl_check_af_backend_option(FL_ARRAYFIRE_USE_CPU ArrayFire::afcpu)

set(_AF_CUDA   ${FL_ARRAYFIRE_USE_CUDA})
set(_AF_CPU    ${FL_ARRAYFIRE_USE_CPU})
set(_AF_OPENCL ${FL_ARRAYFIRE_USE_OPENCL})
if (
    (${_AF_CUDA} AND ${_AF_CPU}) OR
    (${_AF_CPU}  AND ${_AF_OPENCL}) OR
    (${_AF_CUDA} AND ${_AF_OPENCL})
    )
  message(FATAL_ERROR "Multiple FL_ARRAYFIRE_USE_{CUDA,CPU,OPENCL} were "
    "specified as ON. Only one ArrayFire backend can be linked at once.")
endif()

# If all options are off swap options to ON for ArrayFire backend targets
# that were found. Raise an error if multiple targets are found but no options
# were specified as ON
if (NOT (${_AF_CUDA} OR ${_AF_CPU} OR ${_AF_OPENCL}))
  if (
      ((TARGET ArrayFire::afcuda) AND (TARGET ArrayFire::afcpu)) OR
      ((TARGET ArrayFire::afcpu)  AND (TARGET ArrayFire::afopencl)) OR
      ((TARGET ArrayFire::afcuda) AND (TARGET ArrayFire::afopencl))
      )
    message(FATAL_ERROR "Multiple ArrayFire targets were found but no "
      "ArrayFire backend was selected. Exactly one of "
      "FL_ARRAYFIRE_USE_{CUDA,CPU,OPENCL} must be ON.")
  endif()

  # Exactly one ArrayFire target is found, but no options are specified.
  # Set the option based on the found target.
  if (TARGET ArrayFire::afcuda)
    set(FL_ARRAYFIRE_USE_CUDA ON)
  endif()
  if (TARGET ArrayFire::afcpu)
    set(FL_ARRAYFIRE_USE_CPU ON)
  endif()
  if (TARGET ArrayFire::afopencl)
    set(FL_ARRAYFIRE_USE_OPENCL ON)
  endif()
endif()

if (${FL_ARRAYFIRE_USE_CUDA})
  target_link_libraries(flashlight PUBLIC ArrayFire::afcuda)
  fl_set_backend_state(ENABLE CUDA)
elseif(${FL_ARRAYFIRE_USE_CPU})
  target_link_libraries(flashlight PUBLIC ArrayFire::afcpu)
  fl_set_backend_state(ENABLE CPU)
elseif(${FL_ARRAYFIRE_USE_OPENCL})
  target_link_libraries(flashlight PUBLIC ArrayFire::afopencl)
  fl_set_backend_state(ENABLE OPENCL)
endif()

target_compile_definitions(
  flashlight
  PUBLIC
  # AF backends are guaranteed to be found as specified
  FL_ARRAYFIRE_USE_CPU=$<BOOL:${FL_ARRAYFIRE_USE_CPU}>
  FL_ARRAYFIRE_USE_CUDA=$<BOOL:${FL_ARRAYFIRE_USE_CUDA}>
  FL_ARRAYFIRE_USE_OPENCL=$<BOOL:${FL_ARRAYFIRE_USE_OPENCL}>
)

# ----------------------------- Sources -----------------------------

include(${CMAKE_CURRENT_LIST_DIR}/mem/CMakeLists.txt) # memory

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireBinaryOps.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireBLAS.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireReductions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireShapeAndIndex.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireTensor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ArrayFireUnaryOps.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
)

if (${FL_ARRAYFIRE_USE_CPU})
  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/ArrayFireCPUStream.cpp
  )
endif()

if (${FL_ARRAYFIRE_USE_CUDA})
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
