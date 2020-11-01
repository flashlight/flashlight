cmake_minimum_required(VERSION 3.5.1)

add_library(
  experimental_memory
  INTERFACE
)

set(
  MEMORY_MANAGER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/AllocationLog.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/allocator/CompositeMemoryAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/allocator/ConfigurableMemoryAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/allocator/MemoryAllocator.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/allocator/freelist/FreeList.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/allocator/memorypool/MemoryPool.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/optimizer/Optimizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/optimizer/Simulator.cpp
)

target_sources(
  experimental_memory
  INTERFACE
  $<BUILD_INTERFACE:${MEMORY_MANAGER_SOURCES}>
)

target_link_libraries(
  experimental_memory
  INTERFACE
  flashlight
)

add_executable(allocation_log_analyser
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/AllocationLogAnalyser.cpp
  )

target_link_libraries(
  allocation_log_analyser
  PRIVATE
  experimental_memory
  flashlight
  )

target_include_directories(
  allocation_log_analyser
  PRIVATE
  ${CMAKE_SOURCE_DIR}
  )

add_executable(stand_alone_simulator
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/StandAloneSimulator.cpp
)

target_link_libraries(
  stand_alone_simulator
  PRIVATE
  experimental_memory
  flashlight
  )

target_include_directories(
  stand_alone_simulator
  PRIVATE
  ${CMAKE_SOURCE_DIR}
  )


add_executable(stand_alone_optimizer
  ${CMAKE_CURRENT_SOURCE_DIR}/flashlight/fl/experimental/memory/StandAloneOptimizer.cpp
  )

target_link_libraries(
  stand_alone_optimizer
  PRIVATE
  experimental_memory
  flashlight
  )

target_include_directories(
  stand_alone_optimizer
  PRIVATE
  ${CMAKE_SOURCE_DIR}
  )
