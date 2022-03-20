cmake_minimum_required(VERSION 3.10)

# Enable or disable compute backends. Example:
#
#   fl_set_backend_state(ENABLE CUDA DISABLE CPU)
#
# The above indicates a CUDA-compatible backend is being built and should be
# enabled and that CPU backend components should be disabled. Whether to
# build OPENCL backend components will remain unchanged.
function(fl_set_backend_state)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs ENABLE DISABLE)
  cmake_parse_arguments(fl_set_backend_state
    "${options}" "${oneValueArgs}"
    "${multiValueArgs}" ${ARGN} )

  # Enable
  foreach(backend IN ITEMS ${fl_set_backend_state_ENABLE})
    set(FL_USE_${backend} ON PARENT_SCOPE)
  endforeach()
  # Disable
  foreach(backend IN ITEMS ${fl_set_backend_state_DISABLE})
    set(FL_USE_${backend} OFF PARENT_SCOPE)
  endforeach()
endfunction()

# ]--------------- Backend Options and Selection
option(FL_USE_ARRAYFIRE "Build ArrayFire tensor backend" ON)

if (FL_USE_ARRAYFIRE)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/af/CMakeLists.txt)
endif()

target_compile_definitions(
  flashlight
  PUBLIC
  FL_USE_ARRAYFIRE=$<BOOL:${FL_USE_ARRAYFIRE}>
)

# Make sure at least one tensor backend is enabled
if (NOT
    ${FL_USE_ARRAYFIRE}
    )
  message(FATAL_ERROR "Cannot build Flashlight with no tensor backends "
    "enabled. Flashlight must be built with  FL_USE_[backend name] ON "
    "for at least one backend.")
endif()

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Compute.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Index.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Init.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Random.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Shape.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Stream.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorBackend.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorBase.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorAdapter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TensorExtension.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Types.cpp
)

# Profiling -- TODO: move this to runtime things
cmake_dependent_option(FL_BUILD_PROFILING
  "Enable profiling with Flashlight" OFF
  "FL_USE_CUDA" OFF)
if (FL_USE_CUDA)
  target_link_libraries(flashlight PUBLIC ${CUDA_LIBRARIES})
  target_include_directories(flashlight PUBLIC ${CUDA_INCLUDE_DIRS})

  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/CUDAStream.cpp
    ${CMAKE_CURRENT_LIST_DIR}/CUDAUtils.cpp
    )

  if (FL_BUILD_PROFILING)
    # Try to find NVTX
    find_library(CUDA_NVTX_LIBRARIES REQUIRED
      NAMES nvToolsExt
      PATHS "${CUDA_TOOLKIT_ROOT_DIR}"
      ENV CUDA_PATH
      ENV CUDA_LIB_PATH
      ENV CUDA_HOME
      PATH_SUFFIXES lib64 lib
      NO_DEFAULT_PATH
      )

    target_sources(flashlight PRIVATE ${CMAKE_CURRENT_LIST_DIR}/CUDAProfile.cpp)
    target_link_libraries(flashlight PUBLIC ${CUDA_NVTX_LIBRARIES})
  endif()
endif()
