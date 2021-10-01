cmake_minimum_required(VERSION 2.8.8)

# Project name
project(imucalib)

# Include our cmake files
#set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

# Find catkin (the ROS build system)
find_package(catkin QUIET COMPONENTS roscpp)

# Include libraries
find_package(Boost REQUIRED COMPONENTS system filesystem thread date_time)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

# display message to user
message(STATUS "BOOST VERSION: " ${Boost_VERSION})
message(STATUS "EIGEN VERSION: " ${EIGEN3_VERSION})
message(STATUS "CERES VERSION: " ${CERES_VERSION})

# Describe catkin project
if (catkin_FOUND)
    add_definitions(-DROS_AVAILABLE=1)
    catkin_package(
            CATKIN_DEPENDS roscpp
            INCLUDE_DIRS src
            LIBRARIES ${PROJECT_NAME}
    )
else()
    message(WARNING "CATKIN NOT FOUND BUILDING WITHOUT ROS!")
endif()

#[[
# Try to compile with c++11
# http://stackoverflow.com/a/25836953
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()
]]

# Enable compile optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -Wall -Wuninitialized -Wmaybe-uninitialized")

# Include our header files
include_directories(
        src
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
        ${catkin_INCLUDE_DIRS}
)

##################################################
# Make binary files!
##################################################

add_executable(run_record src/run_record.cpp)
target_link_libraries(run_record ${Boost_LIBRARIES} ${CERES_LIBRARIES} ${catkin_LIBRARIES})


#[[
##################################################
# Below is OpenVINS-wrapped codes
##################################################

# Find OpenVINS packages
find_package(catkin QUIET COMPONENTS rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs cv_bridge)
find_package(ov_core REQUIRED)
find_package(ov_msckf REQUIRED)
find_package(ov_eval REQUIRED)
find_package(OpenCV 4 REQUIRED)

# display message to user
message(STATUS "OPENCV VERSION: " ${OpenCV_VERSION})

# Describe catkin project
if (catkin_FOUND)
    add_definitions(-DROS_AVAILABLE=1)
    catkin_package(
            CATKIN_DEPENDS roscpp rosbag tf std_msgs geometry_msgs sensor_msgs nav_msgs visualization_msgs cv_bridge ov_core
            INCLUDE_DIRS src
            LIBRARIES imucalib_lib
    )
else()
    message(WARNING "CATKIN NOT FOUND BUILDING WITHOUT ROS!")
endif()

# Set source directories
# set(ov_core_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../ov_core)
# set(ov_msckf_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../ov_msckf)
# set(ov_eval_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../ov_eval)
# set(ov_calib_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../ov_calib)

# Include our header files
include_directories(
        ${ov_core_SOURCE_DIR}/src
        ${ov_msckf_SOURCE_DIR}/src
        ${ov_eval_SOURCE_DIR}/src
)

# Set link libraries used by all binaries
list(APPEND thirdparty_libraries
        ${Boost_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${catkin_LIBRARIES}
)

# If we are not building with ROS then we need to manually link to its headers
# This isn't that elegant of a way, but this at least allows for building without ROS
# See this stackoverflow answer: https://stackoverflow.com/a/11217008/7718197
if (NOT catkin_FOUND)
    message(WARNING "MANUALLY LINKING TO OV_CORE LIBRARY....")
    include_directories(${ov_core_SOURCE_DIR}/src/)
    list(APPEND thirdparty_libraries ov_core_lib)
endif()



##################################################
# Make simulator library
##################################################
list(APPEND sim_lib_source_files
        ${ov_msckf_SOURCE_DIR}/src/sim/Simulator.cpp
        src/sim/SimulatorMultiIMU.cpp
        src/core/RosTinyVisualizer.cpp
)
add_library(sim_lib SHARED ${sim_lib_source_files})
target_link_libraries(sim_lib ${thirdparty_libraries})
target_include_directories(sim_lib PUBLIC src)
]]