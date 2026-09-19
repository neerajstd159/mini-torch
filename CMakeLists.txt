cmake_minimum_required(VERSION 3.16)

project(
    mini-torch
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Library
# ============================================================

add_library(
    mini_torch_core

    aten/src/ATen/core/Storage.cpp
    aten/src/ATen/core/TensorImpl.cpp

    # torch/csrc/api/src/Tensor.cpp
    torch/csrc/api/src/Creation.cpp
)

target_include_directories(
    mini_torch_core
    PUBLIC

    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/c10
    ${PROJECT_SOURCE_DIR}/aten/src
    ${PROJECT_SOURCE_DIR}/torch/csrc/api/include
)

if(MSVC)
    target_compile_options(mini_torch_core PRIVATE /W4)
else()
    target_compile_options(
        mini_torch_core
        PRIVATE

        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# ============================================================
# TensorImpl example
# ============================================================

add_executable(
    tensor_impl_example

    examples/tensor_impl_example.cpp
)

target_link_libraries(
    tensor_impl_example
    PRIVATE

    mini_torch_core
)

# ============================================================
# Tensor example
# ============================================================

add_executable(
    tensor_example

    examples/tensor_example.cpp
)

target_link_libraries(
    tensor_example
    PRIVATE

    mini_torch_core
)

# ============================================================
# Creation example
# ============================================================

add_executable(
    creation_example

    examples/creation_example.cpp
)

target_link_libraries(
    creation_example
    PRIVATE

    mini_torch_core
)

# ============================================================
# Tests
# ============================================================

enable_testing()

add_executable(
    mini_torch_tests

    test/cpp/test_main.cpp
    test/cpp/test_storage.cpp
    test/cpp/test_tensor_impl.cpp
    test/cpp/test_tensor.cpp
    test/cpp/test_creation.cpp
)

target_link_libraries(
    mini_torch_tests
    PRIVATE

    mini_torch_core
)

add_test(
    NAME mini_torch_tests
    COMMAND mini_torch_tests
)