cmake_minimum_required(VERSION 3.10)

include(${CMAKE_MODULE_PATH}/Buildpybind11.cmake)
include(${CMAKE_MODULE_PATH}/pybind11Tools.cmake)

set_target_properties(
  fl-libraries
  PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE
  TRUE
  )

function (add_pybind11_extension ext_name file_name)
  pybind11_add_module(
    ${ext_name}
    ${CMAKE_CURRENT_LIST_DIR}/${file_name}.cpp
    )

  target_link_libraries(
    ${ext_name}
    PUBLIC
    fl-libraries
    )

  add_dependencies(${ext_name} pybind11)
endfunction ()

add_pybind11_extension(_lib_audio_feature flashlight/lib/audio/_feature)
add_pybind11_extension(_lib_sequence_criterion flashlight/lib/sequence/_criterion)
add_pybind11_extension(_lib_text_decoder flashlight/lib/text/_decoder)
add_pybind11_extension(_lib_text_dictionary flashlight/lib/text/_dictionary)
