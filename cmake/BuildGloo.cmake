cmake_minimum_required(VERSION 3.10.0)

include(ExternalProject)

set(Gloo_TEMP_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern/gloo)
set(Gloo_URL https://github.com/facebookincubator/gloo.git)
set(Gloo_BUILD ${CMAKE_CURRENT_BINARY_DIR}/gloo)
set(Gloo_TAG 1da21174054eaabbbd189b7f657ea24842d821e2)

if (NOT TARGET Gloo)
  # Download Gloo
  ExternalProject_Add(
    Gloo
    PREFIX gloo
    GIT_REPOSITORY ${Gloo_URL}
    GIT_TAG ${Gloo_TAG}
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_COMMAND} --build .
    CMAKE_CACHE_ARGS
      -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${Gloo_TEMP_INSTALL_DIR}
      -DUSE_MPI:BOOL=ON
  )
endif ()

# Install the install executed at build time
install(DIRECTORY ${Gloo_TEMP_INSTALL_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${Gloo_TEMP_INSTALL_DIR}/lib DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${Gloo_TEMP_INSTALL_DIR}/share DESTINATION ${CMAKE_INSTALL_PREFIX})

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
set(gloo_LIBRARIES "${Gloo_TEMP_INSTALL_DIR}/lib/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}gloo${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}")
set(gloo_INCLUDE_DIRS "${Gloo_TEMP_INSTALL_DIR}/include")
# Make dirs so this can be used as an interface include directory
file(MAKE_DIRECTORY ${gloo_INCLUDE_DIRS})

add_library(gloo ${LIB_TYPE} IMPORTED)
set_target_properties(gloo PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${gloo_INCLUDE_DIRS}
  IMPORTED_LOCATION ${gloo_LIBRARIES}
  )
