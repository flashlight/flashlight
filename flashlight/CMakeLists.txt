cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Setup -----------------------------
set(FLASHLIGHT_CORE_COMPONENT_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}") # module root

# ----------------------------- Configuration -----------------------------
# Distributed Training Backend
set(FL_BUILD_DISTRIBUTED ON CACHE BOOL "Whether to build and link the distributed backend with flashlight")
# If building with CUDA, use NCCL to on; if using CPU or OpenCL, use GLOO
set(USE_NCCL FALSE)
set(USE_GLOO FALSE)
if (FL_BUILD_DISTRIBUTED)
  if (FLASHLIGHT_USE_CUDA)
    set(USE_NCCL TRUE)
  elseif (FLASHLIGHT_USE_CPU OR FLASHLIGHT_USE_OPENCL)
    set(USE_GLOO TRUE)
  endif ()
endif ()

# -------------------------------- Components --------------------------------

# Build and link resources from `flashlight/contrib`.
set(FL_BUILD_CONTRIB ON CACHE BOOL
    "Build and link additional flashlight contrib resources.")
if (FL_BUILD_CONTRIB)
  message(STATUS "Will build flashlight contrib assets.")
  include(${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/contrib/CMakeLists.txt)
endif()

# Build and link resources from `flashlight/experimental`.
set(FL_BUILD_EXPERIMENTAL OFF CACHE BOOL
    "Build and link additional internal flashlight experimental resources.")
if (FL_BUILD_EXPERIMENTAL)
  message(STATUS "Will build flashlight experimental assets.")
  set(FL_EXPERIMENTAL_ROOT_PATH
    ${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/experimental)
  include(${FL_EXPERIMENTAL_ROOT_PATH}/CMakeLists.txt)
endif()

# Internal includes are implicitly defined as <flashlight...>
target_include_directories(
  flashlight
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

# -------------------------------- Components --------------------------------

# Common
include(${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/common/CMakeLists.txt)

# Autograd
include(${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/autograd/CMakeLists.txt)

# Dataset
include(${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/dataset/CMakeLists.txt)

# Dataset
include(${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/distributed/CMakeLists.txt)

# Memory
include(${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/memory/CMakeLists.txt)

# Meter
include(${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/meter/CMakeLists.txt)

# NN
include(${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/nn/CMakeLists.txt)

# Optim
include(${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/optim/CMakeLists.txt)

# --------------------------- Configure Examples/Tests ---------------------------

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/test)
endif ()

# Build examples
if (FL_BUILD_EXAMPLES)
  set(FL_EXAMPLES_DIR ${FLASHLIGHT_CORE_COMPONENT_SRC_DIR}/examples)
  add_subdirectory(${FL_EXAMPLES_DIR})
  # Installation properties
  install(
    DIRECTORY ${FL_EXAMPLES_DIR}
    DESTINATION ${FL_INSTALL_EXAMPLES_DIR}
    COMPONENT examples
    )
endif ()
