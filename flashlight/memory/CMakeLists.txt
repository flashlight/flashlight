cmake_minimum_required(VERSION 3.5.1)

add_library(
  Memory
  INTERFACE
)

set(
  MEMORY_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/allocator/MemoryAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/allocator/freelist/FreeList.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/allocator/memorypool/MemoryPool.cpp
)

target_link_libraries(
  Memory
  INTERFACE
  Common
)

target_sources(
  Memory
  INTERFACE
  $<BUILD_INTERFACE:${MEMORY_SOURCES}>
)
