cmake_minimum_required(VERSION 2.8)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
project(4_5_frame_buffer)
find_package(glfw3 REQUIRED)
file(GLOB project_file
	glad.c
	4_5_frame_buffer.cpp)
add_executable(${PROJECT_NAME} ${project_file})

target_link_libraries(
	${PROJECT_NAME}
	-lglfw3
	-lGL
	-lm
	-lXrandr
	-lXi
	-lX11
	-lXxf86vm
	-lpthread
	-ldl
	-lXinerama
	-lXcursor
	)
