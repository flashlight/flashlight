cmake_minimum_required(VERSION 3.5.1)

include(ExternalProject)

set(Gloo_URL https://github.com/facebookincubator/gloo.git)
set(Gloo_BUILD ${CMAKE_CURRENT_BINARY_DIR}/gloo/)
set(Gloo_TAG 1da21174054eaabbbd189b7f657ea24842d821e2) # release 1.10.0

if (NOT TARGET Gloo)
  # Download Gloo
  ExternalProject_Add(
    Gloo
    PREFIX gloo
    GIT_REPOSITORY ${Gloo_URL}
    GIT_TAG ${Gloo_TAG}
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release
    CMAKE_CACHE_ARGS
      -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE:STRING=Release
      -DUSE_MPI:BOOL=ON
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
  )
endif ()

ExternalProject_Get_Property(Gloo source_dir)
set(Gloo_SOURCE_DIR ${source_dir})

ExternalProject_Get_Property(Gloo binary_dir)
set(Gloo_BINARY_DIR ${binary_dir})

if (BUILD_SHARED_LIBS)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

# Library and include dirs
set(gloo_LIBRARIES "${Gloo_BINARY_DIR}/${CMAKE_CFG_INTDIR}/gloo/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}gloo${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}")
set(gloo_INCLUDE_DIRS "${Gloo_BINARY_DIR}/${CMAKE_CFG_INTDIR}")

add_library(gloo ${LIB_TYPE} IMPORTED)
set_target_properties(gloo PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${gloo_INCLUDE_DIRS}
  IMPORTED_LOCATION ${gloo_LIBRARIES}
  )
