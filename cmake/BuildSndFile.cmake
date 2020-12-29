cmake_minimum_required(VERSION 3.10.0)

# SndFile must be built with encoder libs
find_package(Ogg REQUIRED)
find_package(Vorbis REQUIRED)
find_package(FLAC REQUIRED)

include(ExternalProject)

set(SndFile_URL https://github.com/libsndfile/libsndfile.git)
set(SndFile_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sndfile/)
set(SndFile_TEMP_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern/sndfile)
set(SndFile_TAG v1.0.30) # release v1.0.30

if (NOT TARGET SndFile)
  # Download SndFile
  ExternalProject_Add(
    SndFile
    PREFIX sndfile
    GIT_REPOSITORY ${SndFile_URL}
    GIT_TAG ${SndFile_TAG}
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_COMMAND} --build .
    CMAKE_CACHE_ARGS
      -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${SndFile_TEMP_INSTALL_DIR}
      -DBUILD_TESTING:BOOL=OFF
      -DBUILD_PROGRAMS:BOOL=OFF
      -DBUILD_EXAMPLES:BOOL=OFF
      -DBUILD_REGTEST:BOOL=OFF
  )
endif ()

# Install the install executed at build time
install(DIRECTORY ${SndFile_TEMP_INSTALL_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${SndFile_TEMP_INSTALL_DIR}/lib DESTINATION ${CMAKE_INSTALL_PREFIX})
install(DIRECTORY ${SndFile_TEMP_INSTALL_DIR}/share DESTINATION ${CMAKE_INSTALL_PREFIX})

ExternalProject_Get_Property(SndFile source_dir)
set(SndFile_SOURCE_DIR ${source_dir})

ExternalProject_Get_Property(SndFile binary_dir)
set(SndFile_BINARY_DIR ${binary_dir})

if (BUILD_SHARED_LIBS)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

# Library and include dirs
set(SndFile_LIBRARIES "${SndFile_TEMP_INSTALL_DIR}/lib/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}sndfile${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}")
set(SndFile_INCLUDE_DIRS "${SndFile_TEMP_INSTALL_DIR}/include")
# Make dirs so this can be used as an interface include directory
file(MAKE_DIRECTORY ${SndFile_TEMP_INSTALL_DIR})
file(MAKE_DIRECTORY ${SndFile_INCLUDE_DIRS})

get_target_property(VORBIS_LIB Vorbis::vorbis IMPORTED_LOCATION)
get_target_property(VORBIS_ENC_LIB Vorbis::vorbisenc IMPORTED_LOCATION)
get_target_property(FLAC_LIB FLAC::FLAC IMPORTED_LOCATION)
get_target_property(OGG_LIB Ogg::ogg IMPORTED_LOCATION)
list(APPEND SNDFILE_DEP_LIBRARIES
  ${VORBIS_LIB}
  ${VORBIS_ENC_LIB}
  ${FLAC_LIB}
  ${OGG_LIB}
  )

add_library(SndFile::sndfile ${LIB_TYPE} IMPORTED)
set_target_properties(SndFile::sndfile PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${SndFile_INCLUDE_DIRS}"
  IMPORTED_LOCATION "${SndFile_LIBRARIES}"
  INTERFACE_LINK_LIBRARIES "${SNDFILE_DEP_LIBRARIES}"
  )
