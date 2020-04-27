cmake_minimum_required(VERSION 3.5.1)

add_library(
  Memory
  INTERFACE
)

set(
  MEMORY_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/MemoryManagerAdapter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/MemoryManagerInstaller.cpp
  # Managers
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/managers/DefaultMemoryManager.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/managers/CachingMemoryManager.cpp
  # Allocators
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/allocator/MemoryAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/allocator/freelist/FreeList.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/allocator/memorypool/MemoryPool.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/allocator/CompositeMemoryAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/memory/allocator/ConfigurableMemoryAllocator.cpp
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
