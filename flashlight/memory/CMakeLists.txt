cmake_minimum_required(VERSION 3.5.1)

add_library(
  Memory
  INTERFACE
)

set(
  MEMORY_SOURCES
  ${CMAKE_SOURCE_DIR}/flashlight/memory/MemoryManagerAdapter.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/memory/MemoryManagerInstaller.cpp
  # Managers
  ${CMAKE_SOURCE_DIR}/flashlight/memory/managers/DefaultMemoryManager.cpp
  ${CMAKE_SOURCE_DIR}/flashlight/memory/managers/CachingMemoryManager.cpp
)

target_sources(
  Memory
  INTERFACE
  $<BUILD_INTERFACE:${MEMORY_SOURCES}>
)

target_link_libraries(
  Memory
  INTERFACE
  Common
)
