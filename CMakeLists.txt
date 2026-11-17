cmake_minimum_required(VERSION 3.18)
project(vertexcomp LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VC_BUILD_PYTHON "Build the Python module" ON)
option(VC_BUILD_TESTS "Build the C++ test executable" OFF)

if(MSVC)
  add_compile_options(/O2 /fp:precise /utf-8 /W3 /wd4244 /wd4267 /wd4996)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
  add_compile_options(-O3 -Wall -Wno-unused-function -Wno-sign-compare)
endif()

file(GLOB VC_OP_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/core/src/ops/*.cpp)
set(VC_CORE_SOURCES
  core/src/image.cpp
  core/src/threadpool.cpp
  core/src/expr.cpp
  core/src/graph.cpp
  core/src/color.cpp
  core/src/roto.cpp
  core/src/io.cpp
  core/src/stb_impl.c
  ${VC_OP_SOURCES})

# Object library: ops register themselves with static initializers, so they
# must be linked as objects rather than pulled from a static archive.
add_library(vccore OBJECT ${VC_CORE_SOURCES})
target_include_directories(vccore PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core/include)
find_package(Threads REQUIRED)
target_link_libraries(vccore PUBLIC Threads::Threads)
set_source_files_properties(core/src/stb_impl.c PROPERTIES COMPILE_OPTIONS "-w")
if(MSVC)
  set_source_files_properties(core/src/stb_impl.c PROPERTIES COMPILE_OPTIONS "/w")
endif()

if(VC_BUILD_PYTHON)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  if(NOT nanobind_DIR)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  endif()
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_vccore NB_STATIC bindings/module.cpp $<TARGET_OBJECTS:vccore>)
  target_include_directories(_vccore PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core/include)
  target_link_libraries(_vccore PRIVATE Threads::Threads)
  if(SKBUILD)
    install(TARGETS _vccore LIBRARY DESTINATION vertexcomp)
  else()
    # Developer build: drop the module next to the Python package.
    set_target_properties(_vccore PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/app/vertexcomp
                                             LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/app/vertexcomp)
  endif()
endif()

if(VC_BUILD_TESTS)
  add_executable(vc_tests tests/core/test_core.cpp $<TARGET_OBJECTS:vccore>)
  target_include_directories(vc_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core/include ${CMAKE_CURRENT_SOURCE_DIR}/core/src)
  target_link_libraries(vc_tests PRIVATE Threads::Threads)
  enable_testing()
  add_test(NAME vc_tests COMMAND vc_tests)
endif()
