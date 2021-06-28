cmake_minimum_required(VERSION 3.13)

# ----------------------------- Dependencies -----------------------------
# CBLAS
if (FL_LIBRARIES_USE_MKL)
  find_package(MKL REQUIRED)
  set(CBLAS_LIBRARIES ${MKL_LIBRARIES})
  set(CBLAS_INCLUDE_DIR ${MKL_INCLUDE_DIR})
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindMKL.cmake)
else ()
  find_package(CBLAS REQUIRED)
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindCBLAS.cmake)
endif ()
message(STATUS "CBLAS found (include: ${CBLAS_INCLUDE_DIR}, library: ${CBLAS_LIBRARIES})")

# FFTW
find_package(FFTW3 REQUIRED)
if (FFTW3_FOUND)
  message(STATUS "FFTW found")
  setup_install_find_module(${CMAKE_MODULE_PATH}/FindFFTW3.cmake)
else()
  message(FATAL_ERROR "FFTW not found")
endif()

# OpenMP
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Threads/OpenMP pollutes other compilers (e.g. nvcc) with flags
target_compile_options(fl_lib_audio PRIVATE $<$<COMPILE_LANGUAGE:C>:${OpenMP_C_FLAGS}>)
target_compile_options(fl_lib_audio PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
target_compile_options(fl_lib_audio PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_THREAD_LIBS_INIT}>)

target_compile_definitions(
  fl_lib_audio
  PUBLIC
  ${FL_COMPILE_DEFINITIONS}
)

# ----------------------------- Lib -----------------------------
target_sources(
  fl_lib_audio
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Ceplifter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Dct.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Derivatives.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Dither.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Mfcc.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Mfsc.cpp
  ${CMAKE_CURRENT_LIST_DIR}/PowerSpectrum.cpp
  ${CMAKE_CURRENT_LIST_DIR}/PreEmphasis.cpp
  ${CMAKE_CURRENT_LIST_DIR}/SpeechUtils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TriFilterbank.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Windowing.cpp
  )

if (NOT FL_LIBRARIES_USE_MKL)
  # openmp required by PowerSpectrum: provided by iomp if MKL ON
  # Not using the target OpenMP::OpenMP_CXX: it does add fopenmp to nvcc which does not understand it
  set(OMPLIB ${OpenMP_CXX_LIBRARIES})
endif()

target_link_libraries(
  fl_lib_audio
  PUBLIC
    $<BUILD_INTERFACE:FFTW3::fftw3>
  PRIVATE
    ${OMPLIB}
    ${CBLAS_LIBRARIES}
)

target_include_directories(
  fl_lib_audio
  PRIVATE
  ${CBLAS_INCLUDE_DIR}
)
