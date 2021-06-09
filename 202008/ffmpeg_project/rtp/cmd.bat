play rtp stream:
ffplay -protocol_whitelist "file,rtp,udp" play.sdp

read stream type into *.sdp file:
ffmpeg -re -i Sample.h264 -vcodec copy -f rtp rtp://127.0.0.1:1234 > play.sdp