include(ExternalProject)

set(cereal_URL https://github.com/USCiLab/cereal.git)
set(cereal_BUILD ${CMAKE_CURRENT_BINARY_DIR}/cereal)
set(cereal_TAG v1.3.2)

# Download Cereal
ExternalProject_Add(
  cereal
  PREFIX cereal
  GIT_REPOSITORY ${cereal_URL}
  GIT_TAG ${cereal_TAG}
  BUILD_IN_SOURCE 1
  BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release
  INSTALL_COMMAND ""
  CMAKE_CACHE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DJUST_INSTALL_CEREAL:BOOL=ON
    
)
ExternalProject_Get_Property(cereal source_dir)
set(CEREAL_SOURCE_DIR ${source_dir})
ExternalProject_Get_Property(cereal binary_dir)
set(CEREAL_BINARY_DIR ${binary_dir})

# Include dir. dependent on build or install
set(cereal_INCLUDE_DIRS
  $<BUILD_INTERFACE:${CEREAL_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CEREAL_INSTALL_PATH}> # see root CMakeLists.txt
  )
