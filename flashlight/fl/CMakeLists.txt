cmake_minimum_required(VERSION 3.10)

# ----------------------------- Setup -----------------------------
set(FL_CORE_COMPONENT_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}") # module root

# ----------------------------- Configuration -----------------------------
# Distributed Training Backend
cmake_dependent_option(FL_BUILD_DISTRIBUTED
  "Build and link a distributed backend with flashlight" ON
  "FL_BUILD_CORE" OFF)
# If building with CUDA, use NCCL to on; if using CPU or OpenCL, use GLOO
set(USE_NCCL FALSE)
set(USE_GLOO FALSE)
if (FL_BUILD_DISTRIBUTED)
  if (FL_USE_CUDA)
    set(USE_NCCL TRUE)
  elseif (FL_USE_MIOPEN)
    set(USE_RCCL TRUE)
  elseif (FL_USE_CPU OR FL_USE_OPENCL)
    set(USE_GLOO TRUE)
  endif ()
endif ()

# -------------------------------- Components --------------------------------
if (FL_BUILD_CONTRIB)
  message(STATUS "Will build flashlight contrib assets.")
  include(${FL_CORE_COMPONENT_SRC_DIR}/contrib/CMakeLists.txt)
endif()

# Build and link resources from `flashlight/experimental`.
cmake_dependent_option(FL_BUILD_EXPERIMENTAL
  "Build and link additional internal flashlight experimental resources." ON
  "FL_BUILD_CORE" OFF)
if (FL_BUILD_EXPERIMENTAL)
  message(STATUS "Will build flashlight experimental assets.")
  set(FL_EXPERIMENTAL_ROOT_PATH
    ${FL_CORE_COMPONENT_SRC_DIR}/experimental)
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
include(${FL_CORE_COMPONENT_SRC_DIR}/common/CMakeLists.txt)

# Autograd
include(${FL_CORE_COMPONENT_SRC_DIR}/autograd/CMakeLists.txt)

# Dataset
include(${FL_CORE_COMPONENT_SRC_DIR}/dataset/CMakeLists.txt)

# Dataset
include(${FL_CORE_COMPONENT_SRC_DIR}/distributed/CMakeLists.txt)

# Memory
include(${FL_CORE_COMPONENT_SRC_DIR}/memory/CMakeLists.txt)

# Meter
include(${FL_CORE_COMPONENT_SRC_DIR}/meter/CMakeLists.txt)

# NN
include(${FL_CORE_COMPONENT_SRC_DIR}/nn/CMakeLists.txt)

# Optim
include(${FL_CORE_COMPONENT_SRC_DIR}/optim/CMakeLists.txt)

# --------------------------- Configure Examples/Tests ---------------------------

# Build tests
if (FL_BUILD_TESTS)
  add_subdirectory(${FL_CORE_COMPONENT_SRC_DIR}/test)
endif ()

# Build examples
if (FL_BUILD_EXAMPLES)
  set(FL_EXAMPLES_DIR ${FL_CORE_COMPONENT_SRC_DIR}/examples)
  add_subdirectory(${FL_EXAMPLES_DIR})
  # Installation properties
  install(
    DIRECTORY ${FL_EXAMPLES_DIR}
    DESTINATION ${FL_INSTALL_EXAMPLES_DIR}
    COMPONENT examples
    )
endif ()
