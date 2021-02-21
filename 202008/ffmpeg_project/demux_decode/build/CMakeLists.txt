cmake_minimum_required(VERSION 2.8)

PROJECT(FFmpegDraft)

SET(SRC_LIST ../packet_video_encode.c)

include_directories("/usr/local/include")

ADD_EXECUTABLE(packet_video_encode  ${SRC_LIST})

target_link_libraries(packet_video_encode libavformat.so libavcodec.so libavdevice.so libavfilter.so libavutil.so libswscale.so libswresample.so libavutil.so m)
