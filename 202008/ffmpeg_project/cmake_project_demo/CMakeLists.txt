cmake_minimum_required(VERSION 2.8)

PROJECT(ComputerVisionDraft)

SET(SRC_LIST decode_av.c)

include_directories("/usr/local/include")

ADD_EXECUTABLE(ComputerVisionDraft  ${SRC_LIST})

target_link_libraries(ComputerVisionDraft libavformat.so libavcodec.so libavdevice.so libavfilter.so libavutil.so libswscale.so libswresample.so libavutil.so )
