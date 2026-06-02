cmake_minimum_required(VERSION 3.20)
project(chnsw)

set(CMAKE_CXX_STANDARD 20)

file(GLOB_RECURSE SOURCES src/*.cpp src/*.cc)
add_executable(main ${SOURCES})

target_include_directories(main PRIVATE
    src/utils include
)

find_package(Eigen3 CONFIG REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(OpenMP REQUIRED)

target_link_libraries(main PRIVATE Eigen3::Eigen)
if(TARGET HDF5::HDF5)
    target_link_libraries(main PRIVATE HDF5::HDF5)
elseif(TARGET hdf5::hdf5-static)
    target_link_libraries(main PRIVATE hdf5::hdf5-static)
else()
    message(FATAL_ERROR "No supported HDF5 target found")
endif()
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)

if(NOT MSVC)
    target_compile_options(main PRIVATE -march=native)
endif()

file(COPY ${CMAKE_SOURCE_DIR}/data
     DESTINATION ${CMAKE_BINARY_DIR})