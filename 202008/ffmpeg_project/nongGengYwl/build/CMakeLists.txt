cmake_minimum_required(VERSION 2.8)

PROJECT(FFmpegDraft)

SET(SRC_LIST ../frameOverlay.c)

include_directories("/usr/local/include")

ADD_EXECUTABLE(frameOverlay  ${SRC_LIST})

target_link_libraries(frameOverlay libavformat.so libavcodec.so libavdevice.so libavfilter.so libavutil.so libswscale.so libswresample.so libavutil.so m z x264 )
