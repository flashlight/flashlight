cmake_minimum_required(VERSION 3.16)

# backward-cpp
option(FL_USE_BACKWARD_CPP "Build with backward-cpp support" OFF)
if (FL_USE_BACKWARD_CPP)
  find_package(Backward CONFIG)
  if (NOT Backward_FOUND)
    message(STATUS "backward-cpp not found - will download from source.")
    include(${PROJECT_SOURCE_DIR}/cmake/BuildBackwardCpp.cmake)
  endif()
  add_backward(flashlight) # include dirs and compiler defs
  # trace demangler libs
  set_property(TARGET flashlight APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES "${BACKWARD_LIBRARIES}")
endif()
target_sources(flashlight PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Backward.cpp)
target_compile_definitions(flashlight PUBLIC
  FL_USE_BACKWARD_CPP=$<BOOL:${FL_USE_BACKWARD_CPP}>)
