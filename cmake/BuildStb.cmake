include(ExternalProject)

set(stb_URL https://github.com/nothings/stb.git)
set(stb_BUILD ${CMAKE_CURRENT_BINARY_DIR}/stb)
set(stb_TAG b42009b3b9d4ca35bc703f5310eedc74f584be58)

# Download stb
ExternalProject_Add(
  stb
  PREFIX stb
  GIT_REPOSITORY ${stb_URL}
  GIT_TAG ${stb_TAG}
  BUILD_IN_SOURCE 0
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
  LOG_DOWNLOAD ON
)

ExternalProject_Get_Property(stb SOURCE_DIR)
set(stb_SOURCE_DIR ${SOURCE_DIR})

set(stb_INCLUDE_DIRS
  $<BUILD_INTERFACE:${SOURCE_DIR}/>
  $<INSTALL_INTERFACE:${stb_INSTALL_PATH}>
  )
