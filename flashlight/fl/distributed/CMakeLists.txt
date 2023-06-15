cmake_minimum_required(VERSION 3.16)

option(FL_BUILD_DISTRIBUTED
  "Build distributed computation capabilities with Flashlight" ON)

option(FL_USE_NCCL "Build with NCCL for distributed computation" OFF)
option(FL_USE_GLOO "Build with Gloo for distributed computation" OFF)

# TODO: relax this
if (FL_USE_NCCL AND FL_USE_GLOO)
  message(STATUS "Cannot build multiple distributed backends simultaneously")
endif()

# Build option behavior is as follows:
# - if FL_BUILD_DISTRIBUTED is OFF, do nothing and build the distributed
#   stub backend. Build the distributed stub backend if building the
#   tensor stub backend.
# - if FL_BUILD_DISTRIBUTED is ON, defer to other build options to
#   find and build other distributed training implementations:
# - first, try to build the distributed backend(s) that correspond to the
#   base runtime/compute environment (e.g. CUDA). Do so for all backends
#   that are not explicitly disabled (e.g. FL_USE_X=OFF)
if (NOT FL_BUILD_DISTRIBUTED)
  # A stub impl that throws/executes noops for most distributed ops
  set(FL_DISTRIBUTED_STUB ON)
else()
  # If using CUDA, enable NCCL, else build with Gloo
  if (FL_USE_CUDA)
    message(STATUS
      "CUDA support in Flashlight is enabled. Will build with NCCL support.")
    set(FL_USE_NCCL ON)
  elseif (FL_USE_CPU)
    message(STATUS
      "CPU support in Flashlight is enabled. Will build with Gloo support.")
    set(FL_USE_GLOO ON)
  elseif(FL_USE_TENSOR_STUB)
    message(WARNING "Building detected using ONLY the tensor stub backend. "
      "Disabling distributed support and building ONLY the "
      "distributed stub backend.")
    set(FL_BUILD_DISTRIBUTED OFF)
    set(FL_DISTRIBUTED_STUB ON)
  endif()
endif()

# Always build distributed API sources, even if building the stub backend.
target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/DistributedApi.cpp
  ${CMAKE_CURRENT_LIST_DIR}/FileStore.cpp
  ${CMAKE_CURRENT_LIST_DIR}/reducers/InlineReducer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/reducers/CoalescingReducer.cpp
  )

# Builds the dependency-free version of the distributed backend
if (FL_DISTRIBUTED_STUB)
  # Forces dispatch only to the stub backend
  # TODO: relax this when tensor-based dispatch is available
  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/stub/DistributedBackend.cpp
    )
endif()

# ----------------------------- Dependencies -----------------------------
# MPI is required for any distributed support
# TODO: make this and the rendezvous impl configurable
if (FL_BUILD_DISTRIBUTED)
  find_package(MPI REQUIRED)
  target_link_libraries(flashlight PUBLIC MPI::MPI_CXX)
endif ()

if (FL_USE_GLOO)
  find_package(Gloo CONFIG)
  if (NOT Gloo_FOUND)
    if (NOT FL_BUILD_STANDALONE)
      message(FATAL_ERROR "Required dependency Gloo not found")
    endif()
    message(STATUS "Gloo not found - downloading and building from source")
    include(${PROJECT_SOURCE_DIR}/cmake/BuildGloo.cmake)
  endif()

  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/cpu/DistributedBackend.cpp
    )

  target_link_libraries(flashlight PRIVATE gloo)
endif()

if (FL_USE_NCCL)
  find_package(NCCL REQUIRED)
  if (NCCL_FOUND)
    message(STATUS "NCCL found: (include: ${NCCL_INCLUDE_DIRS} lib: ${NCCL_LIBRARIES}")
    setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindNCCL.cmake)
  endif()

  target_sources(
    flashlight
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend/cuda/DistributedBackend.cpp
    )

  target_link_libraries(flashlight PRIVATE ${NCCL_LIBRARIES})
  target_include_directories(flashlight PRIVATE ${NCCL_INCLUDE_DIRS})

  target_compile_definitions(
    flashlight
    PUBLIC
    "-DNO_NCCL_COMM_DESTROY_HANDLE"
    )
endif()
