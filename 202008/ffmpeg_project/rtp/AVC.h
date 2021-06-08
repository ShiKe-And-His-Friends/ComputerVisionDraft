#ifndef RTP_AVC_TAG_H
#define RTP_AVC_TAG_H

#include "typedef.h"

/** copy from FFmpeg libavformat/avc.c */
const uint8_t* ff_avc_find_startcode(const uint8_t* p ,const uint8_t* end);

#endif

