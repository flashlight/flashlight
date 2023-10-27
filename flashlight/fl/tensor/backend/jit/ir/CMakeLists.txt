cmake_minimum_required(VERSION 3.16)

# ----------------------------- Sources -----------------------------
target_sources(
  flashlight
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/BinaryNode.cpp
  ${CMAKE_CURRENT_LIST_DIR}/CustomNode.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ExternalUse.cpp
  ${CMAKE_CURRENT_LIST_DIR}/IndexNode.cpp
  ${CMAKE_CURRENT_LIST_DIR}/IndexedUpdateNode.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Node.cpp
  ${CMAKE_CURRENT_LIST_DIR}/NodeType.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ScalarNode.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Use.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ValueNode.cpp
)
