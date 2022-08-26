/*
    1. MP3 raw data
    2. Process to opencv

    Thanks for your donate : https://zhuanlan.zhihu.com/p/398283696

    create by shike on 20220826
*/

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

extern "C" {
    #include "libavcodec/avcodec.h"
    #include "libavformat/avformat.h"  
}

#pragma comment(lib ,"avcodec.lib")
#pragma comment(lib ,"avformat.lib")
#pragma comment(lib ,"avutil.lib")
#pragma comment(lib ,"swresample.lib")


using namespace std;
using namespace cv;
#define AUDIO_INBUF_SIZE 20480
#define AUDIO_REFILL_THRESH 4096

static int get_format_from_sample_fmt(const char** fmt,
    enum AVSampleFormat sample_fmt)
{
    int i;
    struct sample_fmt_entry {
        enum AVSampleFormat sample_fmt; const char* fmt_be, * fmt_le;
    } sample_fmt_entries[] = {
        { AV_SAMPLE_FMT_U8,  "u8",    "u8"    },
        { AV_SAMPLE_FMT_S16, "s16be", "s16le" },
        { AV_SAMPLE_FMT_S32, "s32be", "s32le" },
        { AV_SAMPLE_FMT_FLT, "f32be", "f32le" },
        { AV_SAMPLE_FMT_DBL, "f64be", "f64le" },
    };
    *fmt = NULL;

    for (i = 0; i < FF_ARRAY_ELEMS(sample_fmt_entries); i++) {
        struct sample_fmt_entry* entry = &sample_fmt_entries[i];
        if (sample_fmt == entry->sample_fmt) {
            *fmt = AV_NE(entry->fmt_be, entry->fmt_le);
            return 0;
        }
    }

    fprintf(stderr,
        "sample format %s error.\n",
        av_get_sample_fmt_name(sample_fmt));
    return -1;
}

static void decode(AVCodecContext* dec_ctx, AVPacket* pkt, AVFrame* frame,
    FILE* outfile)
{
    int i, ch;
    int ret, data_size;

    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "error packet send.\n");
        exit(1);
    }
    
    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "error decoding.\n");
            exit(1);
        }
        data_size = av_get_bytes_per_sample(dec_ctx->sample_fmt);
        if (data_size < 0) {
            fprintf(stderr, "error calculate data size.\n");
            exit(1);
        }
        for (i = 0; i < frame->nb_samples; i++)
            for (ch = 0; ch < dec_ctx->channels; ch++)
                fwrite(frame->data[ch] + data_size * i, 1, data_size, outfile);
    }
}

int main(int argc, char** argv)
{

    cout << avcodec_configuration() << endl;

    const char inFileName[] = "D:\\Aritcle\\music\\menghuanlisha.mp3";
    const char outFileName[] = "D:\\Aritcle\\music\\menghuanlisha.pcm";
    FILE* file = fopen(outFileName, "w+b");
    if (!file) {
        printf("error open file.\n");
        return -1;
    }

    AVFormatContext* fmtCtx = avformat_alloc_context();
    AVCodecContext* codecCtx = NULL;
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    int aStreamIndex = -1;

    do {

        if (avformat_open_input(&fmtCtx, inFileName, NULL, NULL) < 0) {
            printf("error open input file format.\n");
            return -1;
        }
        if (avformat_find_stream_info(fmtCtx, NULL) < 0) {
            printf("error find stream.\n");
            return -1;
        }

        av_dump_format(fmtCtx, 0, inFileName, 0);

        for (size_t i = 0; i < fmtCtx->nb_streams; i++) {
            if (fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                aStreamIndex = (int)i;
                break;
            }
        }
        if (aStreamIndex == -1) {
            printf("error find audio.\n");
            return -1;
        }

        AVCodecParameters* aCodecPara = fmtCtx->streams[aStreamIndex]->codecpar;
        const AVCodec* codec = avcodec_find_decoder(fmtCtx->streams[aStreamIndex]->codecpar->codec_id);
        if (!codec) {
            printf("error find codec audio.\n");
            return -1;
        }
        codecCtx = avcodec_alloc_context3(codec);
        if (avcodec_parameters_to_context(codecCtx, aCodecPara) < 0) {
            printf("error codec context.\n");
            return -1;
        }
        codecCtx->pkt_timebase = fmtCtx->streams[aStreamIndex]->time_base;

        if (avcodec_open2(codecCtx, codec, NULL) < 0) {
            printf("error open audio codec.\n");
            return -1;
        }

        while (av_read_frame(fmtCtx, pkt) >= 0) {
            if (pkt->stream_index == aStreamIndex) {
                if (avcodec_send_packet(codecCtx, pkt) >= 0) {
                    while (avcodec_receive_frame(codecCtx, frame) >= 0) {
                        if (av_sample_fmt_is_planar(codecCtx->sample_fmt)) {
                            int numBytes = av_get_bytes_per_sample(codecCtx->sample_fmt);
                            // PCM格式是LRLRL 要交错保存数据
                            for (int i = 0; i < frame->nb_samples; i++) {
                                for (int ch = 0; ch < codecCtx->channels; ch++) {
                                    fwrite((char*)frame->data[ch] + numBytes * i, 1, numBytes, file);
                                }
                            }
                        }
                    }
                }
            }
            av_packet_unref(pkt);
        }
    } while (0);

    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_close(codecCtx);
    avcodec_free_context(&codecCtx);
    avformat_free_context(fmtCtx);

    fclose(file);

    return 0;
}
