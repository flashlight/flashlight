cmake_minimum_required(VERSION 3.16)

find_package(Protobuf REQUIRED)
find_package(ONNX CONFIG REQUIRED)

target_link_libraries(flashlight PRIVATE onnx)

target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Onnx.cpp
)
