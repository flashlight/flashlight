cmake_minimum_required(VERSION 3.16)

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
option(FL_USE_JIT "Build JIT tensor backend" ON)
option(FL_USE_TENSOR_STUB "Build Stub tensor backend" ON)
option(FL_USE_ONEDNN "Build OneDNN tensor backend" ON)


if (FL_USE_ARRAYFIRE)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/af/CMakeLists.txt)
endif()
if (FL_USE_JIT)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/jit/CMakeLists.txt)
endif()
if (FL_USE_TENSOR_STUB)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/stub/CMakeLists.txt)
endif()
if (FL_USE_ONEDNN)
  include(${CMAKE_CURRENT_LIST_DIR}/backend/onednn/CMakeLists.txt)
endif()

target_compile_definitions(
  flashlight
  PUBLIC
  FL_USE_ARRAYFIRE=$<BOOL:${FL_USE_ARRAYFIRE}>
  FL_USE_JIT=$<BOOL:${FL_USE_JIT}>
  FL_USE_TENSOR_STUB=$<BOOL:${FL_USE_TENSOR_STUB}>
  FL_USE_ONEDNN=$<BOOL:${FL_USE_ONEDNN}>
)

# Make sure at least one tensor backend is enabled
if (NOT (
    ${FL_USE_ARRAYFIRE} OR
    ${FL_USE_TENSOR_STUB} OR
    ${FL_USE_ONEDNN}
    )
  )
  message(FATAL_ERROR "Cannot build Flashlight with no tensor backends "
    "enabled. Flashlight must be built with  FL_USE_[backend name] ON "
    "for at least one backend.")
endif()

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Compute.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DefaultTensorType.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Index.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Init.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Random.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Shape.cpp
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
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
  else()
    message(FATAL_ERROR "CUDA is enabled but no CUDA compiler was found")
  endif()

  # Link CUDA components
  if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    find_package(CUDAToolkit REQUIRED cublas)
    target_link_libraries(flashlight PRIVATE CUDA::cublas)
  else()
    # Remove old branch when requiring CMake >= 3.17
    target_link_libraries(flashlight PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(flashlight PRIVATE ${CUDA_INCLUDE_DIRS})
  endif()

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
