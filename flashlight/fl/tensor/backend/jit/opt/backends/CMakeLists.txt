cmake_minimum_required(VERSION 3.10)

# ----------------------------- Sources -----------------------------
if (FL_USE_ONEDNN)
  include(${CMAKE_CURRENT_LIST_DIR}/onednn/CMakeLists.txt) # onednn
endif()
