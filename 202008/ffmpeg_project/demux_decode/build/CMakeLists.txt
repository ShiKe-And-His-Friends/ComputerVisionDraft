cmake_minimum_required(VERSION 2.8)

PROJECT(FFmpegDraft)

SET(SRC_LIST ../packet_audio_filter_before.c)

include_directories("/usr/local/include")

ADD_EXECUTABLE(packet_audio_filter_before  ${SRC_LIST})

target_link_libraries(packet_audio_filter_before libavformat.so libavcodec.so libavdevice.so libavfilter.so libavutil.so libswscale.so libswresample.so libavutil.so m z)
