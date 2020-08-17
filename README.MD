# Some codes of the Computer Vision Area

## 202008 ffmpeg the offical codes.

### Commands:
***cmake  -D CMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_opencv_xfeatures2d=OFF -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/home/sk95120/Downloads/opencv_contrib-4.4.0/modules  ..***  
***sudo cp -r ./opencv-4.2.0/modules/features2d ./opencv-4.2.0/buildopencv-4.2.0***
***gcc decode_av.c -o decode_av2 -I/usr/local/include -L/usr/local/lib -lavformat -lavcodec -lavdevice -lavfilter -lavutil -lswscale -lswresample -lavutil -lm -lz***

***g++ ffmpeg_demo.cpp -o ffmpeg_demo2 -I/usr/local/include -L/usr/local/lib -lavformat -lavcodec -lavdevice -lavfilter -lavutil -lswscale -lswresample -lavutil -lm -lz***

```C
ret = avcodec_send_packet(dec_ctx, pkt);
```

[MakeDown阅读](www.mdeditor.com) .
