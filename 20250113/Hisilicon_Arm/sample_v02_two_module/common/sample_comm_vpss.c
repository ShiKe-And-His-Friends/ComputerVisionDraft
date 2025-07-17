/*
  Copyright (c), 2001-2024, Shenshu Tech. Co., Ltd.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/time.h>

#include "sample_comm.h"

#define VPSS_DEFAULT_WIDTH  1280
#define VPSS_DEFAULT_HEIGHT 1024

td_void sample_comm_vpss_get_default_grp_attr(ot_vpss_grp_attr *grp_attr)
{
    grp_attr->ie_en                     = TD_FALSE;
    grp_attr->dci_en                    = TD_FALSE;
    grp_attr->buf_share_en              = TD_FALSE;
    //grp_attr->buf_share_en              = TD_TRUE;
    grp_attr->mcf_en                    = TD_FALSE;
    grp_attr->max_width                 = VPSS_DEFAULT_WIDTH;
    grp_attr->max_height                = VPSS_DEFAULT_HEIGHT;
    //grp_attr->max_dei_width             = 0;
    //grp_attr->max_dei_height            = 0;
    grp_attr->max_dei_width             = VPSS_DEFAULT_WIDTH;
    grp_attr->max_dei_height            = VPSS_DEFAULT_HEIGHT;
    grp_attr->dynamic_range             = OT_DYNAMIC_RANGE_SDR8;
    grp_attr->pixel_format              = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_422;
    grp_attr->dei_mode                  = OT_VPSS_DEI_MODE_OFF;
    grp_attr->buf_share_chn             = OT_VPSS_CHN0;
    grp_attr->frame_rate.src_frame_rate = 30;
    grp_attr->frame_rate.dst_frame_rate = 30;
}

td_void sample_comm_vpss_get_default_chn_attr(ot_vpss_chn_attr *chn_attr)
{
    chn_attr->mirror_en                 = TD_FALSE;
    chn_attr->flip_en                   = TD_FALSE;
    chn_attr->border_en                 = TD_FALSE;
    chn_attr->width                     = VPSS_DEFAULT_WIDTH;
    chn_attr->height                    = VPSS_DEFAULT_HEIGHT;
    chn_attr->depth                     = 0;
    chn_attr->chn_mode                  = OT_VPSS_CHN_MODE_USER;
    chn_attr->video_format              = OT_VIDEO_FORMAT_LINEAR;
    chn_attr->dynamic_range             = OT_DYNAMIC_RANGE_SDR8;
    chn_attr->pixel_format              = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    chn_attr->compress_mode             = OT_COMPRESS_MODE_SEG;
    chn_attr->aspect_ratio.mode         = OT_ASPECT_RATIO_NONE;
    chn_attr->frame_rate.src_frame_rate = 30;
    chn_attr->frame_rate.dst_frame_rate = 30;
}

td_void sample_comm_vpss_get_default_3dnr_attr(ot_3dnr_attr *nr_attr)
{
    nr_attr->enable         = TD_FALSE;
    nr_attr->nr_type        = OT_NR_TYPE_VIDEO_NORM;
    nr_attr->compress_mode  = OT_COMPRESS_MODE_FRAME;
    nr_attr->nr_motion_mode = OT_NR_MOTION_MODE_NORM;
}

td_void sample_comm_vpss_get_default_vpss_cfg(sample_sns_type sns_type,sample_vpss_cfg *vpss_cfg, ot_fmu_mode vpss_fmu)
{
    ot_vpss_chn chn;
    ot_size in_size;

    sample_comm_vi_get_size_by_sns_type(sns_type, &in_size);
    sample_comm_vpss_get_default_grp_attr(&vpss_cfg->grp_attr);
    sample_comm_vpss_get_default_3dnr_attr(&vpss_cfg->nr_attr);


    // shikeDebug
    in_size.width = VPSS_DEFAULT_WIDTH;
    in_size.height = VPSS_DEFAULT_HEIGHT;

    vpss_cfg->vpss_grp = 0;
    //vpss_cfg->vpss_grp = 3;
    vpss_cfg->grp_attr.max_width  = in_size.width;
    vpss_cfg->grp_attr.max_height = in_size.height;

    for (chn = 0; chn < OT_VPSS_MAX_PHYS_CHN_NUM; chn++) {
        vpss_cfg->chn_en[chn] = TD_FALSE;
        sample_comm_vpss_get_default_chn_attr(&vpss_cfg->chn_attr[chn]);
        if (chn > OT_VPSS_CHN1) {
            vpss_cfg->chn_attr[chn].compress_mode = OT_COMPRESS_MODE_NONE;
        }
        vpss_cfg->chn_attr[chn].width  = in_size.width;
        vpss_cfg->chn_attr[chn].height = in_size.height;
#ifdef OT_FPGA
        vpss_cfg->chn_attr[chn].frame_rate.src_frame_rate = 30; /* 30: fpga src frame rate */
        vpss_cfg->chn_attr[chn].frame_rate.dst_frame_rate = 30; /* 10: fpga dst frame rate */
#endif
    }
    vpss_cfg->chn_en[OT_VPSS_CHN0] = TD_TRUE;

    if (vpss_fmu == OT_FMU_MODE_WRAP) {
        vpss_cfg->chn_en[1] = TD_TRUE; /* vpss_chn0->vnec & vpss_chn1->vo */
    }
}

static td_s32 sample_common_vpss_set_chn_fmu_mode(ot_vpss_grp grp, ot_vpss_chn chn, td_bool chn0_wrap)
{
    td_s32 ret = TD_SUCCESS;
    if (chn == OT_VPSS_CHN0 && chn0_wrap == TD_TRUE) {
        ret = ss_mpi_vpss_set_chn_fmu_mode(grp, OT_VPSS_CHN0, OT_FMU_MODE_WRAP);
    }
    return ret;
}

static td_s32 sample_common_vpss_start_chn(ot_vpss_grp grp, const td_bool *chn_enable,
    const ot_vpss_chn_attr *chn_attr, td_bool chn0_wrap, td_u32 chn_array_size)
{
    sample_print("shikeDebug vpss start chn set grp_id %d\n", grp);

    ot_vpss_chn vpss_chn;
    td_s32 ret, i;

    for (i = 0; i < (td_s32)chn_array_size; ++i) {
        if (chn_enable[i] == TD_TRUE) {
            vpss_chn = i;
            ret = ss_mpi_vpss_set_chn_attr(grp, vpss_chn, &chn_attr[vpss_chn]);
            if (ret != TD_SUCCESS) {
                sample_print("ss_mpi_vpss_set_chn_attr failed with %#x\n", ret);
                goto disable_chn;
            }
            /* set chn0 wrap fmu mode first, then enable chn */
            ret = sample_common_vpss_set_chn_fmu_mode(grp, vpss_chn, chn0_wrap);
            if (ret != TD_SUCCESS) {
                sample_print("ss_mpi_vpss_set_chn_fmu_mode failed with %#x\n", ret);
                goto disable_chn;
            }
            ret = ss_mpi_vpss_enable_chn(grp, vpss_chn);
            if (ret != TD_SUCCESS) {
                sample_print("ss_mpi_vpss_enable_chn failed with %#x\n", ret);
                goto disable_chn;
            }
            sample_print("shikeDebug vpss start chn enable %d\n", vpss_chn);
        }
    }
    return TD_SUCCESS;

disable_chn:
    for (i = i - 1; i >= 0; i--) {
        if (chn_enable[i] == TD_TRUE) {
            vpss_chn = i;
            ret = ss_mpi_vpss_disable_chn(grp, vpss_chn);
            if (ret != TD_SUCCESS) {
                sample_print("ss_mpi_vpss_disable_chn failed with %#x!\n", ret);
            }
        }
    }
    return TD_FAILURE;
}

td_s32 sample_common_vpss_start(sample_sns_type sns_type,ot_vpss_grp grp, const ot_vpss_grp_attr *grp_attr,
    const sample_vpss_chn_attr *vpss_chn_attr)
{

    sample_print("shikeDebug vpss start grp_id %d\n", grp);

    td_s32 ret;

    if (vpss_chn_attr->chn_array_size < OT_VPSS_MAX_PHYS_CHN_NUM) {
        sample_print("array size(%u) of chn_enable and chn_attr need >= %u!\n",
            vpss_chn_attr->chn_array_size, OT_VPSS_MAX_PHYS_CHN_NUM);
        return TD_FAILURE;
    }

    ret = ss_mpi_vpss_create_grp(grp, grp_attr);
    if (ret != TD_SUCCESS) {
        sample_print("ss_mpi_vpss_create_grp(grp:%d) failed with %#x!\n", grp, ret);
        return TD_FAILURE;
    }

    if (sns_type == FPGA_BT1120_14BIT) {
        //shikeDebug crop 
        ot_vpss_crop_info crop_info;
        crop_info.enable = TD_TRUE;
        crop_info.crop_mode = OT_COORD_ABS;
        crop_info.crop_rect.x = 642;
        crop_info.crop_rect.y = 286;
        crop_info.crop_rect.width = 640;
        crop_info.crop_rect.height = 512;
        ret = ss_mpi_vpss_set_grp_crop(grp, &crop_info);
        if(ret != TD_SUCCESS) {
            sample_print("ss_mpi_vpss_set_grp_crop(grp:%d) failed with %#x!\n", grp, ret);
            return ret;
        }
        sample_print("FPGA_BT1120_14BIT ss_mpi_vpss_set_grp_crop(grp:%d) success.\n", grp);
        ret = ss_mpi_vpss_get_grp_crop(grp, &crop_info);
        if(ret != TD_SUCCESS) {
            sample_print("ss_mpi_vpss_get_grp_crop(grp:%d) failed with %#x!\n", grp, ret);
        return ret;
        }
        sample_print("FPGA_BT1120_14BIT ss_mpi_vpss_get_grp_crop(grp:%d) success.\n", grp);

    } else if (sns_type == COLORCAMERA_MIPIRX_YUV422) {
        ot_vpss_crop_info crop_info;
        crop_info.enable = TD_TRUE;
        crop_info.crop_mode = OT_COORD_ABS;
        crop_info.crop_rect.x = 320;
        crop_info.crop_rect.y = 28;
        crop_info.crop_rect.width = 1280;
        crop_info.crop_rect.height = 1024;
        ret = ss_mpi_vpss_set_grp_crop(grp, &crop_info);
        if(ret != TD_SUCCESS) {
            sample_print("ss_mpi_vpss_set_grp_crop(grp:%d) failed with %#x!\n", grp, ret);
            return ret;
        }
        sample_print("COLORCAMERA_MIPIRX_YUV422 ss_mpi_vpss_set_grp_crop(grp:%d) success.\n", grp);
        ret = ss_mpi_vpss_get_grp_crop(grp, &crop_info);
        if(ret != TD_SUCCESS) {
            sample_print("ss_mpi_vpss_get_grp_crop(grp:%d) failed with %#x!\n", grp, ret);
        return ret;
        }
        sample_print("COLORCAMERA_MIPIRX_YUV422 ss_mpi_vpss_get_grp_crop(grp:%d) success.\n", grp);
    }
    
    ret = ss_mpi_vpss_start_grp(grp);
    if (ret != TD_SUCCESS) {
        sample_print("ss_mpi_vpss_start_grp failed with %#x\n", ret);
        goto destroy_grp;
    }

    ret = sample_common_vpss_start_chn(grp, vpss_chn_attr->chn_enable, &vpss_chn_attr->chn_attr[0],
        vpss_chn_attr->chn0_wrap, OT_VPSS_MAX_PHYS_CHN_NUM);
    if (ret != TD_SUCCESS) {
        goto stop_grp;
    }

    //shikeDebug crop in windows
    // ot_zoom_attr zoom_attr;
    // zoom_attr.enable = TD_TRUE;
    // zoom_attr.mode = OT_COORD_ABS;
    // zoom_attr.rect.x = 642;
    // zoom_attr.rect.y = 286;
    // zoom_attr.rect.width = 640;
    // zoom_attr.rect.height = 512;
    // ret = ss_mpi_vpss_set_grp_zoom_in_window(grp, &zoom_attr);
    // if(ret != TD_SUCCESS) {
    //     sample_print("ss_mpi_vpss_set_grp_zoom_in_window(grp:%d) failed with %#x!\n", grp, ret);
    //     return ret;
    // }
    // sample_print("ss_mpi_vpss_set_grp_zoom_in_window(grp:%d) success.\n", grp);
    // ret = ss_mpi_vpss_get_grp_zoom_in_window(grp, &zoom_attr);
    // if(ret != TD_SUCCESS) {
    //     sample_print("ss_mpi_vpss_get_grp_zoom_in_window(grp:%d) failed with %#x!\n", grp, ret);
    // return ret;
    // }
    // sample_print("ss_mpi_vpss_get_grp_zoom_in_window(grp:%d) success.\n", grp);
    
    return TD_SUCCESS;

stop_grp:
    ret = ss_mpi_vpss_stop_grp(grp);
    if (ret != TD_SUCCESS) {
        sample_print("ss_mpi_vpss_stop_grp failed with %#x!\n", ret);
    }
destroy_grp:
    ret = ss_mpi_vpss_destroy_grp(grp);
    if (ret != TD_SUCCESS) {
        sample_print("ss_mpi_vpss_destroy_grp failed with %#x!\n", ret);
    }
    return TD_FAILURE;
}

td_s32 sample_common_vpss_stop(ot_vpss_grp grp, const td_bool *chn_enable, td_u32 chn_array_size)
{
    td_s32 i;
    td_s32 ret;
    ot_vpss_chn vpss_chn;

    if (chn_array_size < OT_VPSS_MAX_PHYS_CHN_NUM) {
        sample_print("array size(%u) of chn_enable need > %u!\n", chn_array_size, OT_VPSS_MAX_PHYS_CHN_NUM);
        return TD_FAILURE;
    }

    for (i = 0; i < OT_VPSS_MAX_PHYS_CHN_NUM; ++i) {
        if (chn_enable[i] == TD_TRUE) {
            vpss_chn = i;
            ret = ss_mpi_vpss_disable_chn(grp, vpss_chn);
            if (ret != TD_SUCCESS) {
                sample_print("ss_mpi_vpss_disable_chn failed with %#x!\n", ret);
            }
        }
    }

    ret = ss_mpi_vpss_stop_grp(grp);
    if (ret != TD_SUCCESS) {
        sample_print("ss_mpi_vpss_stop_grp failed with %#x!\n", ret);
    }

    ret = ss_mpi_vpss_destroy_grp(grp);
    if (ret != TD_SUCCESS) {
        sample_print("ss_mpi_vpss_destroy_grp failed with %#x!\n", ret);
    }

    return TD_SUCCESS;
}
