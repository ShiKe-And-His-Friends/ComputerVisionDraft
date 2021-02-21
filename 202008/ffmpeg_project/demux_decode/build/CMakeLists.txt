cmake_minimum_required(VERSION 2.8)

PROJECT(FFmpegDraft)

SET(SRC_LIST ../packet_avio.c)

include_directories("/usr/local/include")

ADD_EXECUTABLE(packet_avio  ${SRC_LIST})

target_link_libraries(packet_avio libavformat.so libavcodec.so libavdevice.so libavfilter.so libavutil.so libswscale.so libswresample.so libavutil.so )
