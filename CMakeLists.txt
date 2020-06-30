cmake_minimum_required(VERSION 3.5.1)

project(flashlight)

# --------------------------- Flashlight core ---------------------------

set(FLASHLIGHT_CORE_DIR "${CMAKE_SOURCE_DIR}/flashlight") # module root
add_subdirectory(${FLASHLIGHT_CORE_DIR})

set(FLASHLIGHT_LIB_DIR "${CMAKE_SOURCE_DIR}/libraries") # module root
add_subdirectory(${FLASHLIGHT_LIB_DIR})

set(FLASHLIGHT_EXT_DIR "${CMAKE_SOURCE_DIR}/extensions") # module root
add_subdirectory(${FLASHLIGHT_EXT_DIR})

# --------------------------- Configure Tasks ---------------------------

set(FLASHLIGHT_TASKS_DIR "${CMAKE_SOURCE_DIR}/tasks") # module root
add_subdirectory(${FLASHLIGHT_TASKS_DIR})