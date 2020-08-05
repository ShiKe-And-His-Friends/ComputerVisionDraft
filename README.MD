#Some codes of the Computer Vision Area

##202008 ffmpeg the offical codes.

###Commands:
gcc decode_av.c -o decode_av2 -I/usr/local/include -L/usr/local/lib -lavformat -lavcodec -lavdevice -lavfilter -lavutil -lswscale -lswresample -lavutil -lm -lz

g++ ffmpeg_demo.cpp -o ffmpeg_demo2 -I/usr/local/include -L/usr/local/lib -lavformat -lavcodec -lavdevice -lavfilter -lavutil -lswscale -lswresample -lavutil -lm -lz
