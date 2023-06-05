# Download and build libsox.
# Defines an imported target named sos as well as sox_LIBRARIES and sox_INCLUDE_DIRS
#
# Use:
#  include(${path}/BuildSox.cmake)
#  add_dependencies(target sox)
#  target_link_libraries(target ${sox_LIBRARIES})
#  target_include_directories(target ${sox_INCLUDE_DIRS})
#
# based on
# https://github.com/pytorch/audio/blob/2c8aad97fc8d7647ee8b2df2de9312cce0355ef6/third_party/sox/CMakeLists.txt
# Without opusfile due to version incompatbility errors.
#
cmake_minimum_required(VERSION 3.10.0)

include(ExternalProject)

if (BUILD_SHARED_LIBS)
  set(LIB_TYPE SHARED)
  set(LIB_TYPE_ARGS --enable-shared --disable-static)
else()
  set(LIB_TYPE STATIC)
  set(LIB_TYPE_ARGS --disable-shared --enable-static)
endif()

set(Sox_TEMP_INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/third-party/sox)
set(Sox_ARCHIVE_DIR ${Sox_TEMP_INSTALL_DIR}/archives)
set(Sox_CONFIG_DIR ${CMAKE_CURRENT_BINARY_DIR}/sox/src)
set(COMMON_ARGS --quiet ${LIB_TYPE_ARGS} --prefix=${Sox_TEMP_INSTALL_DIR} --with-pic --disable-dependency-tracking)

# To pass custom environment variables to ExternalProject_Add command,
# we need to do `${CMAKE_COMMAND} -E env ${envs} <COMMANAD>`.
# https://stackoverflow.com/a/62437353
# We constrcut the custom environment variables here
set(envs
  "PKG_CONFIG_PATH=${Sox_TEMP_INSTALL_DIR}/lib/pkgconfig"
  "LDFLAGS=-L${Sox_TEMP_INSTALL_DIR}/lib $ENV{LDFLAGS}"
  "CFLAGS=-I${Sox_TEMP_INSTALL_DIR}/include -fvisibility=hidden $ENV{CFLAGS}"
)

set(SOX_OPTIONS
  --disable-openmp
  --without-amrnb
  --without-amrwb
  --without-lame
  --without-mad
  --without-flac
  --without-oggvorbis
  --without-opus
  --without-alsa
  --without-ao
  --without-coreaudio
  --without-oss
  --without-id3tag
  --without-ladspa
  --without-magic
  --without-png
  --without-pulseaudio
  --without-sndfile
  --without-sndio
  --without-sunaudio
  --without-waveaudio
  --without-twolame
  )

set(Sox_LIBRARIES ${Sox_TEMP_INSTALL_DIR}/lib/libsox.a)

ExternalProject_Add(Sox
  PREFIX sox
  DOWNLOAD_DIR ${Sox_ARCHIVE_DIR}
  URL https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.bz2
  URL_HASH SHA256=81a6956d4330e75b5827316e44ae381e6f1e8928003c6aa45896da9041ea149c
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${envs} ${Sox_CONFIG_DIR}/Sox/configure ${COMMON_ARGS} ${SOX_OPTIONS}
  BUILD_BYPRODUCTS ${Sox_LIBRARIES}
  BUILD_IN_SOURCE 1
  DOWNLOAD_NO_PROGRESS OFF
  LOG_DOWNLOAD ON
  LOG_UPDATE ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  LOG_INSTALL ON
  LOG_MERGED_STDOUTERR ON
  LOG_OUTPUT_ON_FAILURE ON
)

install(DIRECTORY ${Sox_TEMP_INSTALL_DIR}/lib DESTINATION ${CMAKE_INSTALL_PREFIX})

set(SOX_FOUND 1)

ExternalProject_Get_Property(Sox SOURCE_DIR)
set(Sox_SOURCE_DIR ${SOURCE_DIR})

ExternalProject_Get_Property(Sox BINARY_DIR)
set(Sox_BINARY_DIR ${BINARY_DIR})

# Library and include dirs
set(sox_LIBRARIES ${Sox_LIBRARIES})
set(sox_INCLUDE_DIRS "${Sox_TEMP_INSTALL_DIR}/include")
file(MAKE_DIRECTORY ${sox_INCLUDE_DIRS})

set(LIB_FILE "${Sox_TEMP_INSTALL_DIR}/lib/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}sox${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}")

add_library(sox ${LIB_TYPE} IMPORTED)
add_dependencies(sox Sox)
set_target_properties(sox PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${sox_INCLUDE_DIRS}
  IMPORTED_LOCATION ${LIB_FILE}
  )
