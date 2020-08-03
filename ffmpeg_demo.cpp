#include <iostream>
using namespace std;

#define __STDC_CONSTANT_MACROS  

extern "C" {
	#include "libavcodec/avcodec.h"
	#include "libavformat/avformat.h"
	#include "libavutil/avutil.h" 
	//#include "libavutil/opt.h"
	//#include "libswscale/swscale.h"
}

int main(int argc, char *argv[])
{
	cout<<"FFmpeg Test!"<<endl;
	
	av_register_all();
	
	return 0;
}
