/**
 * Camera stream push in FFmpeg
 * https://blog.csdn.net/zhoubotong2012/article/details/102774983
 */

#include "libavformat/avformat.h"  
#include "libavformat/avio.h"  
#include "libavcodec/avcodec.h"  
#include "libswscale/swscale.h"  
#include "libavutil/avutil.h"  
#include "libavutil/mathematics.h"  
#include "libswresample/swresample.h"  
#include "libavutil/opt.h"  
#include "libavutil/channel_layout.h"  
#include "libavutil/samplefmt.h"  
#include "libavdevice/avdevice.h"  //摄像头所用  
#include "libavfilter/avfilter.h"  
#include "libavutil/error.h"  
#include "libavutil/mathematics.h"    
#include "libavutil/time.h"    
#include "inttypes.h"  
#include "stdint.h"
 
static int interruptCallBack(void *ctx)
{
    FileStreamPushTask * pSession = (FileStreamPushTask*) ctx;
 
   //once your preferred time is out you can return 1 and exit from the loop
    if(pSession->CheckTimeOut(GetTickCount()))
    {
      return 1;
    }
 
   //continue 
   return 0;
 
}

BOOL   FileStreamPushTask::CheckTimeOut(DWORD dwCurrentTime)
{
	if(dwCurrentTime < m_dwLastSendFrameTime) //CPU时间回滚
	{
		return FALSE;
	}
 
	if(m_stop_status)
		return TRUE;
 
	if(m_bInited)
	{
		if(m_dwLastSendFrameTime > 0)
		{
			if((dwCurrentTime - m_dwLastSendFrameTime) > 15000) //发送过程中超时
			{
				return TRUE;
			}
		}
	}
	else
	{
		if((dwCurrentTime - m_dwStartConnectTime) > 5000) //连接超时
		{
			TRACE("Connect timeout! \n");
			m_stop_status = true;
			return TRUE;
		}
	}
	return FALSE;
}
 
int main (int argc ,char* argv[]) {
	
	AVFormatContext* m_outputAVFormatCxt;
	/**
	 * RTSP push: res = avformat_alloc_output_context2(&m_outputAVFormatCxt, NULL, "rtsp", m_outputUrl.c_str());
	 * RTMP push: res = avformat_alloc_output_context2(&m_outputAVFormatCxt, NULL, "flv", m_outputUrl.c_str());
	 */
	res = avformat_alloc_output_context2(&m_outputAVFormatCxt, NULL, "flv", argv[1]);
	AVOutputFormat* fmt = m_outputAVFormatCxt->oformat;
 
	//  fmt->video_codec = AV_CODEC_ID_H264;
	//	fmt->audio_codec = AV_CODEC_ID_AAC;
 
	for (int i = 0; i < m_inputAVFormatCxt->nb_streams; i++)
	{
		AVStream *in_stream = m_inputAVFormatCxt->streams[i];
 
		if(in_stream->codec->codec_type != AVMEDIA_TYPE_VIDEO && in_stream->codec->codec_type != AVMEDIA_TYPE_AUDIO)  //忽略掉不是视频和音频的流
		{  
			continue;  
		} 
 
		AVStream *out_stream = avformat_new_stream(m_outputAVFormatCxt, in_stream->codec->codec);
		if (!out_stream)
		{
			TRACE("can not new out stream");
		}
		res = avcodec_copy_context(out_stream->codec, in_stream->codec);
		if (res < 0)
		{
			string strError = "can not copy context, filepath: " + m_filePath + ",errcode:" + to_string(res) + ",err msg:" + av_make_error_string(m_tmpErrString, AV_ERROR_MAX_STRING_SIZE, res);
			TRACE("%s \n", argv[1]);
		}
 
		out_stream->codec->codec_tag = 0; 
		if (m_outputAVFormatCxt->oformat->flags & AVFMT_GLOBALHEADER)
		{
			out_stream->codec->flags |= CODEC_FLAG_GLOBAL_HEADER;
		}
	}
	
	//network busy callback
	m_outputAVFormatCxt->flags |= AVFMT_FLAG_NONBLOCK;
 
	av_dump_format(m_outputAVFormatCxt, 0, m_outputUrl.c_str(), 1);
	if (!(fmt->flags & AVFMT_NOFILE))
	{	
		AVIOInterruptCB icb = {interruptCallBack,this};
 
		m_dwStartConnectTime = GetTickCount();
 
	   // res = avio_open(&m_outputAVFormatCxt->pb, m_outputUrl.c_str(), AVIO_FLAG_WRITE);
		res = avio_open2(&m_outputAVFormatCxt->pb, m_outputUrl.c_str(), AVIO_FLAG_WRITE, &icb, NULL);
		if (res < 0)
		{
			string strError = "can not open output io, URL:" + m_outputUrl;
			TRACE("%s \n", strError.c_str());
			return FALSE;
		}
	}
	
	// connect network
	AVDictionary* options = NULL;
	if(bIsRTSP)
	    av_dict_set(&options, "rtsp_transport", "tcp", 0); 
	av_dict_set(&options, "stimeout", "8000000", 0);  //设置超时时间  
 
	res = avformat_write_header(m_outputAVFormatCxt, &options);
 
	TRACE("avformat_write_header() return: %d\n", res);
 
	if (res < 0)
	{
		string strError = "can not write outputstream header, URL:" + m_outputUrl + ",errcode:" + to_string(res);
		TRACE("%s \n", strError.c_str());
		m_bInited = FALSE;
		return FALSE;
	}
	m_bInited = TRUE;
	
	//write stream and change timstamp
	if(in_stream->codec->codec_type == AVMEDIA_TYPE_VIDEO)  //视频
	{
		if(pkt.pts == AV_NOPTS_VALUE) //没有时间戳
		{
			AVRational time_base1 = out_stream->time_base; 
 
			//Duration between 2 frames (us) 
			int64_t calc_duration =(double)AV_TIME_BASE/av_q2d(in_stream->r_frame_rate); 
 
			pkt.pts = (double)(nVideoFramesNum*calc_duration)/(double)(av_q2d(time_base1)*AV_TIME_BASE); 
			pkt.dts = pkt.pts; 
			pkt.duration = (double)calc_duration/(double)(av_q2d(time_base1)*AV_TIME_BASE); 
		}
		else
		{
			pkt.pts = av_rescale_q_rnd(pkt.pts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
			pkt.dts = av_rescale_q_rnd(pkt.dts, in_stream->time_base, out_stream->time_base, (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
			pkt.duration = av_rescale_q(pkt.duration, in_stream->time_base, out_stream->time_base);
			pkt.pos = -1;
		}
 
		nVideoFramesNum++;
	}
	// write the compressed frame to the output format
	int nError = av_interleaved_write_frame(m_outputAVFormatCxt, &pkt);
	
	//close stream pusher	
	/**
		int res = av_write_trailer(m_outputAVFormatCxt); 
		if (!(m_outputAVFormatCxt->oformat->flags & AVFMT_NOFILE))
		{
			if(m_outputAVFormatCxt->pb)
			{
				avio_close(m_outputAVFormatCxt->pb);
				m_outputAVFormatCxt->pb = nullptr;
			}
		}
	*/	
	
	//hand shark error ,close stream pubshlisher way'
	if (m_outputAVFormatCxt)
	{
		if(m_bInited)
		{
		  int res = av_write_trailer(m_outputAVFormatCxt); 
		}
 
		if (!(m_outputAVFormatCxt->oformat->flags & AVFMT_NOFILE))
		{
			if(m_outputAVFormatCxt->pb)
			{
				avio_close(m_outputAVFormatCxt->pb);
				m_outputAVFormatCxt->pb = nullptr;
			}
		}
 
		avformat_free_context(m_outputAVFormatCxt);
		m_outputAVFormatCxt = nullptr;
	}

	return 0;
}