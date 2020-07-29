cmake_minimum_required(VERSION 3.10)

# give a name to the project
project(imageRectification)

# find opencv
set(OpenCV_DIR "/home/auv/software/opencv/build_cuda10.0" )
find_package(OpenCV REQUIRED)

# check opencv version
if(${OpenCV_VERSION} VERSION_LESS 3.0.0)
  message(FATAL_ERROR “OpenCV version is not compatible : ${OpenCV_VERSION}”)
endif()

# include eigen
find_package(Eigen3 REQUIRED)
if(${EIGEN3_FOUND})
  message(STATUS "lib EIGEN3 found")
  message(STATUS "  version " ${EIGEN3_VERSION_STRING})
  message(STATUS "  include " ${EIGEN3_INCLUDE_DIR})
endif()

# compilation flags
set(CMAKE_CXX_FLAGS "-Wall -g -O2 -std=c++11 -Wno-unused-function")

# the files to compile
add_executable(imageRectification src/main.cpp src/PointCorrespondanceLoader.hpp src/PointCorrespondanceLoader.cpp src/LevenbergMarquardt.cpp src/LevenbergMarquardt.hpp src/MultiRectification.cpp src/MultiRectification.hpp)

# dependecy
include_directories(${JPEG_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# specify the dependencies  // important for make install
target_link_libraries(imageRectification ${OpenCV_LIBS} gflags)




