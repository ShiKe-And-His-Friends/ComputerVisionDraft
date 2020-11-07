# FFmpeg 4.2 

# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Define project name
project(ffmpeg_stream_pull_project)

SET(SRC_LIST ../info.hpp)
SET(SRC_LIST ../server_or_local_pull.hpp)
SET(SRC_LIST ../server_or_local_pull.cpp)
SET(SRC_LIST ../server_or_local_pull_main.cpp)

# Declare the executable target built from your sources
add_executable(ffmpeg_stream_pull_project ${SRC_LIST})

# Link your application with OpenCV libraries
target_link_libraries(ffmpeg_stream_pull_project 
	libavformat.so 
	libavcodec.so 
	libavdevice.so 
	libavfilter.so 
	libavutil.so 
	libswscale.so 
	libswresample.so 
	libavutil.so 
)
