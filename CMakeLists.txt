cmake_minimum_required(VERSION 3.18)
project(dotnxs LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE cpp/src/native_writer.cpp)
target_compile_features(_core PRIVATE cxx_std_17)
target_compile_definitions(_core PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)

install(TARGETS _core
    LIBRARY DESTINATION dotnxs
    RUNTIME DESTINATION dotnxs
)
