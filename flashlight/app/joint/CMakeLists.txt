cmake_minimum_required(VERSION 3.10)

# ------------------------ Joint Trainer ------------------------
add_library(
  flashlight-app-joint
  ""
  )

target_sources(
  flashlight-app-joint
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Flags.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Trainer.cpp
  )

target_link_libraries(
  flashlight-app-joint
  PUBLIC
  ${GLOG_LIBRARIES}
  ${gflags_LIBRARIES}
  flashlight
  flashlight-app-asr
  flashlight-app-lm
  fl-libraries
  )

target_include_directories(
  flashlight-app-joint
  PUBLIC
  ${GLOG_INCLUDE_DIRS}
  ${gflags_INCLUDE_DIRS}
  )

# ------------------------ Binary ------------------------
add_executable(fl_joint_train ${CMAKE_CURRENT_LIST_DIR}/JointTrain.cpp)

target_link_libraries(fl_joint_train flashlight-app-joint)

set_executable_output_directory(fl_joint_train "${FL_BUILD_BINARY_OUTPUT_DIR}")

install(TARGETS fl_joint_train RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})