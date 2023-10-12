cmake_minimum_required(VERSION 3.16)

project(flashlight LANGUAGES CXX C VERSION 0.4.0)

include(CTest)
include(CMakeDependentOption)

# ----------------------------- Setup -----------------------------
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The CUDA standard is still C++14 to enable interopability with
# slightly older and still well-supported versions of CUDA/nvcc
# (e.g. CUDA < 11). This will be bumped to 17 once CUDA 11 is
# required.
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

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
  if (NOT WIN32) # Windows operates with explicit dll exports
    message(WARNING
      "This compiler doesn't support dynamic symbol exports. "
      "Plugin functionality likely won't work.")
  endif()
endif()
set(FL_BUILD_PLUGIN ${COMPILER_SUPPORTS_RDYNAMIC})

# ]------ The library target for the Flashlight core
add_library(flashlight)

# We haven't [yet] polluted Flashlight with explicit dllexport annotations
if (MSVC)
  target_compile_options(flashlight PUBLIC $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/bigobj>)
endif()
if (WIN32)
  target_compile_definitions(flashlight PUBLIC _USE_MATH_DEFINES NOMINMAX)
endif()

include(InternalUtils)

# ----------------------------- Configuration -----------------------------

option(FL_BUILD_TESTS "Build tests" ON)
option(FL_CODE_COVERAGE "Enable coverage reporting" OFF)
option(FL_BUILD_EXAMPLES "Build examples" ON)
option(FL_BUILD_EXPERIMENTAL "Build internal experimental components" OFF)
option(FL_BUILD_SCRIPTS "Build internal scripts for wav2letter++" OFF)
option(FL_BUILD_RECIPES "Build recipes" ON)
option(FL_BUILD_STANDALONE "Build standalone installation" ON)

if (FL_BUILD_TESTS)
  enable_testing()
  include(TestUtils)
endif()

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

# --------------------------- Core ---------------------------
# Internal includes are implicitly defined as <flashlight...>
target_include_directories(
  flashlight
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
)

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
  FL_BUILD_PROFILING=$<BOOL:${FL_BUILD_PROFILING}>
  FL_COMPILE_LIBRARY=$<BOOL:${BUILD_SHARED_LIBS}>
  PRIVATE
  FL_DLL
  )

if (FL_CODE_COVERAGE)
  add_coverage_to_target(TARGET flashlight)
endif()

# --------------------------- pkg/app ---------------------------
include(${FL_ROOT_DIR}/pkg/CMakeLists.txt)
include(${FL_ROOT_DIR}/app/CMakeLists.txt)

# --------------------------- Cleanup ---------------------------
setup_install_targets(INSTALL_TARGETS ${INSTALLABLE_TARGETS})
