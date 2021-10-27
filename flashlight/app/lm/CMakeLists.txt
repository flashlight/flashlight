cmake_minimum_required(VERSION 3.10)

# ----------------------------- Binaries -----------------------------
add_executable(fl_lm_train
  ${CMAKE_CURRENT_LIST_DIR}/Train.cpp
)
target_sources(fl_lm_train
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Trainer.cpp
)
add_executable(fl_lm_test ${CMAKE_CURRENT_LIST_DIR}/Test.cpp)

target_sources(fl_lm_test
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/Trainer.cpp
)
add_executable(
  fl_lm_dictionary_builder
  ${CMAKE_CURRENT_LIST_DIR}/BuildDictionary.cpp
  )

target_link_libraries(fl_lm_train fl_pkg_text fl_pkg_runtime)
target_link_libraries(fl_lm_test fl_pkg_text fl_pkg_runtime)
target_link_libraries(fl_lm_dictionary_builder fl_pkg_text fl_pkg_runtime)

set_executable_output_directory(fl_lm_train "${FL_BUILD_BINARY_OUTPUT_DIR}/lm")
set_executable_output_directory(fl_lm_test "${FL_BUILD_BINARY_OUTPUT_DIR}/lm")
set_executable_output_directory(
  fl_lm_dictionary_builder
  "${FL_BUILD_BINARY_OUTPUT_DIR}/lm"
  )

install(TARGETS fl_lm_train RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
install(TARGETS fl_lm_test RUNTIME DESTINATION ${FL_INSTALL_BIN_DIR})
install(
  TARGETS
  fl_lm_dictionary_builder
  RUNTIME
  DESTINATION
  ${FL_INSTALL_BIN_DIR}
  )
