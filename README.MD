## Example Codes Of The Computer Vision Area
  
### Official websit Link
FFmpeg http://ffmpeg.org/

OpenCv https://docs.opencv.org/

OpenGL http://www.opengl.cn/

WebRtc http://sites.google.com.sixxs.org/site/webrtc/home

PCL  http://www.pclcn.org/

MATLAB  https://www.mathworks.com/products/matlab.html

Hisilicon  https://www.hisilicon.com/cn/products/smart-vision/public-security/IPC/Hi3519DV500

### Commands Lines:

示例1，OpenCv的项目组建：
```C
cmake  -D CMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_opencv_xfeatures2d=OFF -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/home/sk95120/Downloads/opencv_contrib-4.4.0/modules  ..
sudo cp -r ./opencv-4.2.0/modules/features2d ./opencv-4.2.0/buildopencv-4.2.0
```

示例2，C文件的编译命令：
```C
gcc decode_av.c -o decode_av2 -I/usr/local/include -L/usr/local/lib -lavformat -lavcodec -lavdevice -lavfilter -lavutil -lswscale -lswresample -lavutil -lm -lz
```
示例3，C++文件的编译命令：
```C
g++ ffmpeg_demo.cpp -o ffmpeg_demo2 -I/usr/local/include -L/usr/local/lib -lavformat -lavcodec -lavdevice -lavfilter -lavutil -lswscale -lswresample -lavutil -lm -lz
```

示例4，一处FFMPEG调用：
```C
./configure  --enable-shared  --enable-static --enable-gpl --enable-libx264 --enable-libfreetype --prefix=/usr/local/ffmpeg
ret = avcodec_send_packet(dec_ctx, pkt);
```


### Note：
&nbsp;&nbsp;In 2019-2021, I working at a company about online meeting software. After some real project practice. I can make that media encode/decode skilled at Android machine. Also camera preview and video data draws.

&nbsp;&nbsp;For a long time, I confused that how meida data processing. I read book and search solvtion and see public codes.

&nbsp;&nbsp;FFmpeg is a basic media project. OpenCv is a basic project that process picture. OpenGL is a basic hardware rendering language. I speed my night time and "free time" what practices above codes.

&nbsp;&nbsp;In 2022-2024, I worked at a machine vision company about industrial defect detection. I learn to using opencv PCL and matlab. My leader is a women phd who did not like my work method and my educational backgroud. I also learn to using 3D laser camera and 3D reconstruction.

&nbsp;&nbsp;In 2025 so far, I worked at a infraed vision company about middle infrared(MWRI) HgCdTd FPGA device.I using matlab to do algorithm simulation. Meanwhile I do some ARM job in Hisilicon embedded system.

[MakeDown阅读](www.mdeditor.com) .
