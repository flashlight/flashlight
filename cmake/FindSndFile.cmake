# Try to find libsndfile
#
# Provides the cmake config target
# - SndFile::sndfile
#
# Inputs:
#   SndFile_INC_DIR: include directory for sndfile headers
#   SndFile_LIB_DIR: directory containing sndfile libraries
#   SndFile_ROOT_DIR: directory containing sndfile installation
#
# Defines:
#  SndFile_FOUND - system has libsndfile
#  SndFile_INCLUDE_DIRS - the libsndfile include directory
#  SndFile_LIBRARIES - Link these to use libsndfile
#

find_package(SndFile CONFIG)

if (TARGET SndFile::sndfile AND NOT SndFile_WITH_EXTERNAL_LIBS)
  message(FATAL_ERROR "Found sndfile but was NOT built with external libs.")
endif()

if (NOT TARGET SndFile::sndfile)
  # SndFile must be built with encoder libs. These transitive
  # dependencies will be taken care of if a SndFile config is found via
  # find_dependency() - we need them for creating a new IMPORTED target
  find_package(Ogg REQUIRED)
  find_package(Vorbis REQUIRED)
  find_package(FLAC REQUIRED)

  find_path(
    SndFile_INCLUDE_DIR
      sndfile.h
    PATHS
    ${SndFile_INC_DIR}
    ${SndFile_ROOT_DIR}/include
    PATH_SUFFIXES
    include
    )

  find_library(
    SndFile_LIBRARY
    sndfile
    PATHS
    ${SndFile_LIB_DIR}
    ${SndFile_ROOT_DIR}
    PATH_SUFFIXES
    lib
    HINTS
    SNDFILE
    )

  set(SndFile_INCLUDE_DIRS
    ${SndFile_INCLUDE_DIR}
    )
  set(SndFile_LIBRARIES
    ${SndFile_LIBRARY}
    )

  mark_as_advanced(SndFile_INCLUDE_DIRS SndFile_LIBRARIES)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(SndFile DEFAULT_MSG SndFile_INCLUDE_DIRS SndFile_LIBRARIES)

  if (SndFile_FOUND)
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
    
    add_library(SndFile::sndfile UNKNOWN IMPORTED)
    set_target_properties(SndFile::sndfile PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${SndFile_INCLUDE_DIRS}"
      IMPORTED_LOCATION "${SndFile_LIBRARIES}"
      INTERFACE_LINK_LIBRARIES "${SNDFILE_DEP_LIBRARIES}"
      )
    message(STATUS "Found libsndfile: (lib: ${SndFile_LIBRARIES} include: ${SndFile_INCLUDE_DIRS}")
  else()
    message(STATUS "libsndfile not found.")
  endif()
endif() # NOT TARGET SndFile::sndfile
