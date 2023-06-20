cmake_minimum_required(VERSION 3.16)

option(FL_BUILD_DISTRIBUTED
  "Build distributed computation capabilities with Flashlight" ON)

# Determine build option defaults
if (APPLE OR NOT FL_BUILD_DISTRIBUTED)
  # If FL_BUILD_DISTRIBUTED is OFF (i.e. it was explicitly disabled since
  # it defaults to ON), don't build appropriate distributed backends
  # nccl has no macOS support (no CUDA)
  set(_build_nccl_default FALSE)
  # no gloo on macOS (https://github.com/facebookincubator/gloo/issues/262)
  set(_build_gloo_default FALSE)
else()
  # Default building distributed backends to ON if using corresponding compute
  # environment is enabled
  set(_build_nccl_default ${FL_USE_CUDA})
  set(_build_gloo_default ${FL_USE_CPU})
endif()

option(FL_USE_NCCL "Build with NCCL for distributed computation" ${_build_nccl_default})
option(FL_USE_GLOO "Build with Gloo for distributed computation" ${_build_gloo_default})

# TODO: relax this
if (FL_USE_NCCL AND FL_USE_GLOO)
  message(STATUS "Cannot build multiple distributed backends simultaneously")
endif()

# If no distributed backend is enabled, don't build distributed (build stub)
if (FL_BUILD_DISTRIBUTED AND NOT (FL_USE_NCCL OR FL_USE_GLOO))
  message(WARNING "FL_BUILD_DISTRIBUTED is enabled but no distributed"
    " backend was selected. Setting FL_BUILD_DISTRIBUTED to OFF.")
  set(FL_BUILD_DISTRIBUTED OFF)
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
