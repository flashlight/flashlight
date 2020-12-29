cmake_minimum_required(VERSION 3.10)

project(flashlight LANGUAGES CXX C)

include(CTest)
include(CMakeDependentOption)

# ----------------------------- Setup -----------------------------

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
option(FL_BUILD_LIBRARIES "Build flashlight libraries" ON)
option(FL_BUILD_CORE "Build flashlight core" ON)
cmake_dependent_option(FL_BUILD_CONTRIB
  "Build and link additional flashlight contrib assets." ON
  "FL_BUILD_CORE" OFF)
cmake_dependent_option(FL_BUILD_APPS
  "Build flashlight apps" ON
  "FL_BUILD_CORE;FL_BUILD_CONTRIB;FL_BUILD_LIBRARIES" OFF)

# Flashlight backend
set(FL_BACKEND "CUDA" CACHE STRING "Backend with which to build flashlight")
# Select from exactly one backend
set_property(CACHE FL_BACKEND PROPERTY STRINGS CPU CUDA OPENCL)
# Map to flags
set(FL_USE_CPU OFF)
set(FL_USE_CUDA OFF)
set(FL_USE_OPENCL OFF)
if (FL_BACKEND STREQUAL "CPU")
  set(FL_USE_CPU ON)
elseif (FL_BACKEND STREQUAL "CUDA")
  set(FL_USE_CUDA ON)
elseif (FL_BACKEND STREQUAL "OPENCL")
  set(FL_USE_OPENCL ON)
else ()
  message(FATAL_ERROR "Invalid FLASHLIGHT backend specified")
endif ()

# List of installable targets
set(INSTALLABLE_TARGETS "")

set(FL_ROOT_DIR ${PROJECT_SOURCE_DIR}/flashlight)
set(FL_BUILD_BINARY_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")

# CUDA/nvcc setup
if (FL_USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  include(${CMAKE_MODULE_PATH}/CUDAUtils.cmake)
  set_cuda_arch_nvcc_flags()

  # remove me? add -fPIC to nvcc flags if needed
  if (FL_LIBRARIES_BUILD_FOR_PYTHON OR CMAKE_POSITION_INDEPENDENT_CODE)
    cuda_enable_position_independent_code()
  endif ()
endif()

# ------------------------ Tests ------------------------

if (FL_BUILD_TESTS)
  enable_testing()
  include(TestUtils)
endif ()

# ------------------------ flashlight Libraries ------------------------
# FIXME: libraries are required to build core tests - https://git.io/JL955
if (FL_BUILD_LIBRARIES OR (FL_BUILD_CORE AND FL_BUILD_TESTS))
  message(STATUS "Will build flashlight libraries.")

  option(FL_LIBRARIES_USE_CUDA "Use CUDA in flashlight libraries build" ON)
  option(FL_LIBRARIES_USE_KENLM "Use KenLM in flashlight libraries build" ON)
  option(FL_LIBRARIES_USE_MKL "Use MKL in flashlight libraries build" ON)
  option(FL_LIBRARIES_BUILD_FOR_PYTHON "Build Python bindings" OFF)

  set(FL_COMPILE_DEFINITIONS
    $<$<BOOL:${FL_LIBRARIES_USE_CUDA}>:FL_LIBRARIES_USE_CUDA>
    $<$<BOOL:${FL_LIBRARIES_USE_MKL}>:FL_LIBRARIES_USE_MKL>
    $<$<BOOL:${FL_LIBRARIES_USE_KENLM}>:FL_LIBRARIES_USE_KENLM>
    )

  set(FL_LIB_DIR "${FL_ROOT_DIR}/lib")
  include(${FL_LIB_DIR}/CMakeLists.txt)

  if (FL_LIBRARIES_BUILD_FOR_PYTHON)
    set(FL_BINDING_PYTHON "${PROJECT_SOURCE_DIR}/bindings/python")
    include(${FL_BINDING_PYTHON}/CMakeLists.txt)
  endif ()

  setup_install_headers(${FL_LIB_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC})
endif() # if FL_BUILD_LIBRARIES

# --------------------------- flashlight ---------------------------
if (FL_BUILD_CORE)
  message(STATUS "Will build flashlight core and extensions.")

  # Primary lib target. All non-library code is linked here, including app
  add_library(flashlight "")

  set_target_properties(
    flashlight
    PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14
    )

  # FIXME: libraries are required to build core tests - https://git.io/JL955
  if (FL_BUILD_LIBRARIES OR (FL_BUILD_CORE AND FL_BUILD_TESTS))
    target_link_libraries(flashlight PUBLIC fl-libraries)
  endif()

  # ------------------------ Global External Dependencies ------------------------
  # ArrayFire
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

  # Check the proper ArrayFire backend is present
  if (FL_USE_CPU AND NOT ArrayFire_CPU_FOUND)
    message(FATAL_ERROR "ArrayFire CPU not found: cannot build CPU backend")
  elseif (FL_USE_CUDA AND NOT ArrayFire_CUDA_FOUND)
    message(FATAL_ERROR "ArrayFire CUDA not found: cannot build CUDA backend")
  elseif (FL_USE_OPENCL AND NOT ArrayFire_OpenCL_FOUND)
    message(FATAL_ERROR "ArrayFire OpenCL not found: cannot build OpenCL backend")
  endif()

  # If cereal is found in a user-defined location, use it rather than
  # downloading from source
  find_package(cereal)
  if (NOT TARGET cereal AND NOT cereal_FOUND AND FL_BUILD_STANDALONE)
    message(STATUS "cereal NOT found. Will download from source")
    set(CEREAL_INSTALL_PATH ${FL_INSTALL_INC_DIR}/cereal)
    include(${CMAKE_MODULE_PATH}/BuildCereal.cmake)
    add_dependencies(flashlight cereal)
    # Move cereal headers at install time
    install(DIRECTORY ${CEREAL_SOURCE_DIR}/include/cereal
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

  # -------------------- Locate Backend-specific Dependencies --------------------
  # TODO: rather than conditionally searching for backend-specific dependencies,
  # always search for all dependencies, and dynamically build all backends for
  # which all required dependencies are found.

  if (FL_USE_CUDA)
     find_package(CUDA 9.2 QUIET) # CUDA 9.2 is required for >= ArrayFire 3.6.1
     if (CUDA_FOUND)
       message(STATUS "CUDA found (library: ${CUDA_LIBRARIES} include: ${CUDA_INCLUDE_DIRS})")
     else()
       message(STATUS "CUDA not found")
       message(FATAL_ERROR "CUDA required to build CUDA backend")
     endif()
  endif()

  if (FL_USE_OPENCL)
    find_package(OpenCL)
    if (OpenCL_FOUND)
      message(STATUS "OpenCL found (library: ${OpenCL_LIBRARIES} include: ${OpenCL_INCLUDE_DIRS})")
    else()
      message(STATUS "OpenCL not found")
      if (FL_USE_OPENCL)
        message(FATAL_ERROR "OpenCL required to build OpenCL backend")
      endif ()
    endif()
  endif()

  # Set up variables to link the right ArrayFire backend
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

  # ------------------------ flashlight Core ------------------------
  set(FL_CORE_DIR "${FL_ROOT_DIR}/fl")
  include(${FL_CORE_DIR}/CMakeLists.txt)
  set(INSTALLABLE_TARGETS ${INSTALLABLE_TARGETS} flashlight)
  # flashlight core components keep their relative paths with respect to project
  setup_install_headers(${FL_CORE_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC})
  target_compile_definitions(flashlight
    PUBLIC
    FL_BACKEND_CPU=$<BOOL:${FL_USE_CPU}>
    FL_BACKEND_CUDA=$<BOOL:${FL_USE_CUDA}>
    FL_BACKEND_OPENCL=$<BOOL:${FL_USE_OPENCL}>
    )

  # ------------------------ Extensions ------------------------
  set(FL_EXT_DIR "${FL_ROOT_DIR}/ext")
  include(${FL_EXT_DIR}/CMakeLists.txt)

  # Make #include "flashlight/lib" work by adding to the inner incl dir
  setup_install_headers(${FL_EXT_DIR} ${FL_INSTALL_INC_DIR_HEADER_LOC})
endif() # if FL_BUILD_CORE

# --------------------------- Apps ---------------------------
if (FL_BUILD_APPS)
  message(STATUS "Will build Flashlight applications.")
  set(FL_APPS_DIR "${FL_ROOT_DIR}/app")
  include(${FL_APPS_DIR}/CMakeLists.txt)
endif()

# --------------------------- Cleanup ---------------------------
setup_install_targets(INSTALL_TARGETS ${INSTALLABLE_TARGETS})
