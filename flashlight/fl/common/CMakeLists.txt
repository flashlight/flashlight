cmake_minimum_required(VERSION 3.16)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Utils.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DevicePtr.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Defines.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DynamicBenchmark.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Logging.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Histogram.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Plugin.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Timer.cpp
)

# A native threading library
find_package(Threads REQUIRED)
target_link_libraries(flashlight PUBLIC Threads::Threads)

# Dynamic lib loading needed for Plugins
target_link_libraries(flashlight PUBLIC ${CMAKE_DL_LIBS})

# std::filesystem -- remove this when requiring gcc >= 9
find_package(Filesystem REQUIRED COMPONENTS Final Experimental)
setup_install_find_module(${PROJECT_SOURCE_DIR}/cmake/FindFilesystem.cmake)
target_link_libraries(flashlight PUBLIC std::filesystem)
