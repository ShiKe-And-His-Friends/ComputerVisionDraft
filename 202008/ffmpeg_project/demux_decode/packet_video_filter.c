#define _XOPEN_SOURCE 600 /** for thread usleep **/
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <linavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <liavutil/opt.h>

const char *filter_descr = "scale=78:24,transpose=cclock"
/** other way:
 *  scale=78:24[scl]:[scl] transpose=cclock 
 *  //assumes "[in]" and "[out]" to be input output pads respectively
 * **/



