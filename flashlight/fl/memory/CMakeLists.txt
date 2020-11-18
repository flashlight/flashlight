cmake_minimum_required(VERSION 3.10)

set(
  MEMORY_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/MemoryManagerAdapter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/MemoryManagerInstaller.cpp
  # Managers
  ${CMAKE_CURRENT_LIST_DIR}/managers/DefaultMemoryManager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/managers/CachingMemoryManager.cpp
)

target_sources(
  flashlight
  PRIVATE
  ${MEMORY_SOURCES}
)
