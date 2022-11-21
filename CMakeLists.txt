cmake_minimum_required(VERSION 3.16)

project(flashlight LANGUAGES CXX C VERSION 0.4.0)

include(CTest)
include(CMakeDependentOption)

# ----------------------------- Setup -----------------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FL_CODE_COVERAGE "Enable coverage reporting" OFF)

# Default directories for installation
set(FL_INSTALL_INC_DIR "include" CACHE PATH "Install path for headers")
set(FL_INSTALL_INC_DIR_HEADER_LOC ${FL_INSTALL_INC_DIR}/flashlight)
set(FL_INSTALL_LIB_DIR "lib" CACHE PATH "Install path for libraries")
set(FL_INSTALL_BIN_DIR "bin" CACHE PATH "Install path for binaries")
# Other assets
set(FL_INSTALL_ASSETS_BASE_DIR "share/flashlight")
set(FL_INSTALL_CMAKE_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/cmake" CACHE PATH "Install path for CMake files")
set(FL_INSTALL_EXAMPLES_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/examples" CACHE PATH "Install path for example files")
set(FL_INSTALL_DOC_DIR "${FL_INSTALL_ASSETS_BASE_DIR}/doc" CACHE PATH "Install path for documentation")

include(CheckCXXCompilerFlag)
# All libraries should have their symbols exported so plugins can lazily
# symbols from any of them
check_cxx_compiler_flag("-rdynamic" COMPILER_SUPPORTS_RDYNAMIC)
if(${COMPILER_SUPPORTS_RDYNAMIC})
  message(STATUS "-rdynamic supported.")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
else()
  message(WARNING
    "This compiler doesn't support dynamic symbol exports. "
    "Plugin functionality likely won't work.")
endif()

include(InternalUtils)

# ----------------------------- Configuration -----------------------------

option(FL_BUILD_TESTS "Build tests" ON)
option(FL_BUILD_EXAMPLES "Build examples" ON)
option(FL_BUILD_EXPERIMENTAL "Build internal experimental components" OFF)
option(FL_BUILD_SCRIPTS "Build internal scripts for wav2letter++" OFF)
option(FL_BUILD_RECIPES "Build recipes" ON)
option(FL_BUILD_STANDALONE "Build standalone installation" ON)

# External project configuration
option(FL_TEXT_REQUIRE_KENLM "Require KenLM in the Flashlight text build" OFF)

# List of installable targets
set(INSTALLABLE_TARGETS)

set(FL_ROOT_DIR ${PROJECT_SOURCE_DIR}/flashlight)
set(FL_BUILD_BINARY_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")

# ]--- Backend Options
# TODO: change these to FL_BACKEND_* to be consistent with preproc
option(FL_USE_CUDA   "Build CUDA support for Flashlight backends" OFF)
option(FL_USE_CPU    "Build CPU support for Flashlight backends"  OFF)
option(FL_USE_OPENCL "Build OpenCL support for Flashlight backends" OFF)
option(FL_USE_MKL    "Build MKL support for Flashlight backends" OFF)

# -------------------- Locate Backend-specific Dependencies --------------------
find_package(CUDA 9.2 QUIET) # CUDA 9.2 is required for >= ArrayFire 3.6.1
if (CUDA_FOUND)
  message(STATUS "CUDA found (library: ${CUDA_LIBRARIES} include: ${CUDA_INCLUDE_DIRS})")
endif()

find_package(OpenCL)
if (OpenCL_FOUND)
  message(STATUS "OpenCL found (library: ${OpenCL_LIBRARIES} include: ${OpenCL_INCLUDE_DIRS})")
endif()

# CUDA/nvcc setup
if (CUDA_FOUND)
  enable_language(CUDA)
  # The CUDA standard is still C++14 to enable interopability with
  # slightly older and still well-supported versions of CUDA/nvcc
  # (e.g. CUDA < 11). This will be bumped to 17 once CUDA 11 is
  # required.
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  include(${CMAKE_MODULE_PATH}/CUDAUtils.cmake)
  set_cuda_arch_nvcc_flags()

  # remove me when upgrading CMake versions
  # (i.e. add -fPIC to nvcc flags if needed)
  if (CMAKE_POSITION_INDEPENDENT_CODE)
    cuda_enable_position_independent_code()
  endif ()
endif()

# ------------------------ Tests ------------------------

if (FL_BUILD_TESTS)
  enable_testing()
  include(TestUtils)
endif ()

# --------------------------- Core Options  ---------------------------

option(FL_BUILD_CORE "Build flashlight core" ON)

fl_dependent_option(
  FL_BUILD_CONTRIB
  "Build and link additional flashlight contrib assets."
  ON "FL_BUILD_CORE" OFF)

# --------------------------- flashlight ---------------------------
if (FL_BUILD_CORE)
  message(STATUS "Will build flashlight core and extensions.")

  # Primary lib target. All non-library code is linked here, including app
  add_library(flashlight "")

  set_target_properties(
    flashlight
    PROPERTIES
    LINKER_LANGUAGE CXX
    )

  if (FL_CODE_COVERAGE)
    add_coverage_to_target(TARGET flashlight)
  endif()

  # ------------------------ Global External Dependencies ------------------------
  # If cereal is found in a user-defined location, use it rather than
  # downloading from source
  find_package(cereal)
  if (NOT TARGET cereal AND NOT cereal_FOUND AND FL_BUILD_STANDALONE)
    message(STATUS "cereal NOT found. Will download from source")
    set(CEREAL_INSTALL_PATH ${FL_INSTALL_INC_DIR}/cereal)
    include(${CMAKE_MODULE_PATH}/BuildCereal.cmake)
    add_dependencies(flashlight cereal)
    # Move cereal headers at install time
    install(DIRECTORY ${CEREAL_SOURCE_DIR}/include/cereal/
      DESTINATION ${CEREAL_INSTALL_PATH}
      COMPONENT cereal
      FILES_MATCHING
      PATTERN "*.hpp"
      PATTERN "*.h"
      PATTERN ".git" EXCLUDE
      )
    install(FILES ${CEREAL_SOURCE_DIR}/LICENSE ${CEREAL_SOURCE_DIR}/README.md
      DESTINATION ${CEREAL_INSTALL_PATH}
      )
    target_include_directories(flashlight PUBLIC ${cereal_INCLUDE_DIRS})
  else()
    message(STATUS "Found cereal")
    target_link_libraries(flashlight PRIVATE cereal)
  endif()
  setup_install_find_module(${CMAKE_MODULE_PATH}/Findcereal.cmake)

  # ------------------------ flashlight Core ------------------------
  set(FL_CORE_DIR "${FL_ROOT_DIR}/fl")
  include(${FL_CORE_DIR}/CMakeLists.txt)
  list(APPEND INSTALLABLE_TARGETS flashlight)
  # flashlight core components keep their relative paths with respect to project
  setup_install_headers(${FL_CORE_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC})
  # FL_USE_* variables are set in flashlight/fl/tensor/CMakeLists.txt based on
  # backend registration
  target_compile_definitions(flashlight
    PUBLIC
    FL_BACKEND_CPU=$<BOOL:${FL_USE_CPU}>
    FL_BACKEND_CUDA=$<BOOL:${FL_USE_CUDA}>
    FL_BACKEND_OPENCL=$<BOOL:${FL_USE_OPENCL}>
    FL_BUILD_PROFILING=$<BOOL:${FL_BUILD_PROFILING}>
    )

  # Check external dependencies match used backend state. If we're required
  # to link to a backend and it wasn't found, fatal.
  if (FL_USE_CUDA AND NOT CUDA_FOUND)
    message(FATAL_ERROR
      "One or more tensor backends require CUDA to build. CUDA was not found.")
  endif()
  if (FL_USE_OPENCL AND NOT OpenCL_FOUND)
    message(FATAL_ERROR
      "One or more tensor backends require OpenCL to build. OpenCL was not found.")
  endif()

endif() # if FL_BUILD_CORE

# --------------------------- Pkgs  ---------------------------
set(FL_PKG_DIR "${FL_ROOT_DIR}/pkg")
include(${FL_PKG_DIR}/CMakeLists.txt)

# --------------------------- Apps Options  ---------------------------
set(FL_APPS_DIR "${FL_ROOT_DIR}/app")
include(${FL_APPS_DIR}/CMakeLists.txt)

# --------------------------- Cleanup ---------------------------
setup_install_targets(INSTALL_TARGETS ${INSTALLABLE_TARGETS})
