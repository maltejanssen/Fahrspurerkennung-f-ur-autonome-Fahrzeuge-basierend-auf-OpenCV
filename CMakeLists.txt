cmake_minimum_required(VERSION 3.15)
project(lineDetection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(lineDetection ${SOURCE_FILES})

find_package(OpenCV REQUIRED)

target_link_libraries(lineDetection ${OpenCV_LIBS})

#include_directories("C:\\\\opencv\\\\opencv3.6\\\\opencv-opencv-59f0319\\\\build\\\\install\\\\include")

#set(TARGET_LIBS
#        "C:\\\\opencv\\\\opencv3.6\\\\opencv-opencv-59f0319\\\\build\\\\install\\\\x64\\\\mingw\\\\lib\\\\libopencv_core412.dll.a"
#        "C:\\\\opencv\\\\opencv3.6\\\\opencv-opencv-59f0319\\\\build\\\\install\\\\x64\\\\mingw\\\\lib\\\\libopencv_videoio412.dll.a"
#        "C:\\\\opencv\\\\opencv3.6\\\\opencv-opencv-59f0319\\\\build\\\\install\\\\x64\\\\mingw\\\\lib\\\\libopencv_highgui412.dll.a"
#        )


#target_link_libraries(lineDetection ${TARGET_LIBS})



#add_executable(scratch scratch.cpp)