cmake_minimum_required(VERSION 3.16)

# ----------------------------- Dependencies -----------------------------
# BLAS
find_package(MKL)
if (MKL_FOUND)
  set(FL_USE_MKL ON)
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindMKL.cmake)
  set(CBLAS_LIBRARIES ${MKL_LIBRARIES})
  set(CBLAS_INCLUDE_DIR ${MKL_INCLUDE_DIR})
  # TODO: remove me when we consolidate build options
  target_compile_definitions(fl_pkg_speech PUBLIC FL_USE_MKL=$<BOOL:${FL_USE_MKL}>)
else()
  find_package(CBLAS REQUIRED)
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindCBLAS.cmake)
endif()
message(STATUS "CBLAS found (include: ${CBLAS_INCLUDE_DIR}, library: ${CBLAS_LIBRARIES})")
target_include_directories(fl_pkg_speech PRIVATE ${CBLAS_INCLUDE_DIR})
target_link_libraries(fl_pkg_speech PRIVATE ${CBLAS_LIBRARIES})

# FFTW
find_package(FFTW3 REQUIRED)
if (FFTW3_FOUND)
  message(STATUS "FFTW found")
  setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindFFTW3.cmake)
else()
  message(FATAL_ERROR "FFTW not found")
endif()
target_link_libraries(fl_pkg_speech PUBLIC $<BUILD_INTERFACE:FFTW3::fftw3>)

# OpenMP
if (NOT MKL_FOUND)
  # NB: MKL provides iomp if enabled
  find_package(OpenMP REQUIRED)
  target_link_libraries(fl_pkg_speech PRIVATE OpenMP::OpenMP_CXX)
endif()

# Threads
find_package(Threads REQUIRED)
target_link_libraries(fl_pkg_speech PRIVATE Threads::Threads)

# ----------------------------- Lib -----------------------------
target_sources(
  fl_pkg_speech
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
