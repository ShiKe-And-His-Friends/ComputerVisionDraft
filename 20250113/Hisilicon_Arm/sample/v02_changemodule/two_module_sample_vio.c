/*
  Copyright (c), 2001-2024, Shenshu Tech. Co., Ltd.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <limits.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "sample_comm.h"
#include "sample_ipc.h"
#include "securec.h"
#include "ss_mpi_ae.h"

#define X_ALIGN 16
#define Y_ALIGN 2
#define out_ratio_1(x) ((x) / 3)
#define out_ratio_2(x) ((x) * 2 / 3)
#define out_ratio_3(x) ((x) / 2)

static volatile sig_atomic_t g_sig_flag = 0;

/* this configuration is used to adjust the size and number of buffer(VB).  */
static sample_vb_param g_vb_param = {
    .vb_size = {1920, 1080},
    //.pixel_format =  {OT_PIXEL_FORMAT_RGB_BAYER_12BPP, OT_PIXEL_FORMAT_YVU_SEMIPLANAR_422},
    //.pixel_format =  {OT_PIXEL_FORMAT_YVU_SEMIPLANAR_422 ,OT_PIXEL_FORMAT_YVU_SEMIPLANAR_422},
    .pixel_format =  {OT_PIXEL_FORMAT_YVU_SEMIPLANAR_422 ,OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420},
    .compress_mode = {OT_COMPRESS_MODE_SEG, OT_COMPRESS_MODE_SEG},
    .video_format = {OT_VIDEO_FORMAT_LINEAR, OT_VIDEO_FORMAT_LINEAR},
    .blk_num = {16, 16}
};

static sampe_sys_cfg g_vio_sys_cfg = {
    .route_num = 1,
    .mode_type = OT_VI_OFFLINE_VPSS_OFFLINE,
    .nr_pos = OT_3DNR_POS_VI,
    .vi_fmu = {0},
    .vpss_fmu = {0},
};

static sample_vo_cfg g_vo_cfg = {
    .vo_dev            = SAMPLE_VO_DEV_UHD,
    .vo_layer          = SAMPLE_VO_LAYER_VHD0,
    .vo_intf_type      = OT_VO_INTF_MIPI,
    .intf_sync         = OT_VO_OUT_1080P30,
    .bg_color          = COLOR_RGB_BLACK,
    .pix_format        = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420,
    .disp_rect         = {0, 0, 1920, 1080},
    .image_size        = {1920, 1080},
    .vo_part_mode      = OT_VO_PARTITION_MODE_SINGLE,
    .dis_buf_len       = 3, /* 3: def buf len for single */
    .dst_dynamic_range = OT_DYNAMIC_RANGE_SDR8,
    .vo_mode           = VO_MODE_1MUX,
    .compress_mode     = OT_COMPRESS_MODE_SEG,
};

sample_mipi_tx_config g_mipi_tx_config = {
    .intf_sync = OT_MIPI_TX_OUT_1080P30,
};

static sample_comm_venc_chn_param g_venc_chn_param = {
    .frame_rate           = 30, /* 30 is a number */
    .stats_time           = 2,  /* 2 is a number */
    .gop                  = 60, /* 60 is a number */
    .venc_size            = {1920, 1080},
    .size                 = -1,
    .profile              = 0,
    .is_rcn_ref_share_buf = TD_FALSE,
    .gop_attr             = {
        .gop_mode = OT_VENC_GOP_MODE_NORMAL_P,
        .normal_p = {2},
    },
    .type                 = OT_PT_H265,
    //.rc_mode              = SAMPLE_RC_CBR,
    .rc_mode              = SAMPLE_RC_FIXQP,
    
};

static td_u32 sample_vio_get_fmu_wrap_num(ot_fmu_mode fmu_mode[], td_u32 len)
{
    printf("shikeDebug  vio run get fmu warp.\n");
    td_u32 i;
    td_u32 cnt = 0;

    for (i = 0; i < len; i++) {
        if (fmu_mode[i] == OT_FMU_MODE_WRAP) {
            cnt++;
        }
    }
    return cnt;
}

static td_s32 sample_vio_fmu_wrap_init(sampe_sys_cfg *fmu_cfg, ot_size *in_size)
{
    printf("shikeDebug  vio run fmu wrap init\n");

    td_u32 cnt;
    ot_fmu_attr fmu_attr;

    cnt = sample_vio_get_fmu_wrap_num(fmu_cfg->vi_fmu, fmu_cfg->route_num);
    if (cnt > 0) {
        fmu_attr.wrap_en = TD_TRUE;
        fmu_attr.page_num = MIN2(ot_common_get_fmu_wrap_page_num(OT_FMU_ID_VI,
            in_size->width, in_size->height) + (cnt - 1) * 3, /* 3: for multi pipe */
            OT_FMU_MAX_Y_PAGE_NUM);
    } else {
        fmu_attr.wrap_en = TD_FALSE;
    }
    if (ss_mpi_sys_set_fmu_attr(OT_FMU_ID_VI, &fmu_attr) != TD_SUCCESS) {
        return TD_FAILURE;
    }

    cnt = sample_vio_get_fmu_wrap_num(fmu_cfg->vpss_fmu, fmu_cfg->route_num);
    if (cnt > 0) {
        fmu_attr.wrap_en = TD_TRUE;
        fmu_attr.page_num = MIN2(ot_common_get_fmu_wrap_page_num(OT_FMU_ID_VPSS,
            in_size->width, in_size->height) + (cnt - 1) * 3, /* 3: for multi pipe */
            OT_FMU_MAX_Y_PAGE_NUM + OT_FMU_MAX_C_PAGE_NUM);
    } else {
        fmu_attr.wrap_en = TD_FALSE;
    }
    if (ss_mpi_sys_set_fmu_attr(OT_FMU_ID_VPSS, &fmu_attr) != TD_SUCCESS) {
        return TD_FAILURE;
    }

    return TD_SUCCESS;
}

/* define SAMPLE_MEM_SHARE_ENABLE, when use tools to dump YUV/RAW. */
#ifdef SAMPLE_MEM_SHARE_ENABLE
td_void sample_vio_init_mem_share(td_void)
{
    printf("shikeDebug  vio run int mem share.\n");
    td_u32 i;
    ot_vb_common_pools_id pools_id = {0};

    if (ss_mpi_vb_get_common_pool_id(&pools_id) != TD_SUCCESS) {
        sample_print("get common pool_id failed!\n");
        return;
    }
    for (i = 0; i < pools_id.pool_cnt; ++i) {
        ss_mpi_vb_pool_share_all(pools_id.pool[i]);
    }
}
#endif

static td_s32 sample_vio_sys_init(td_void)
{
    printf("shikeDebug  vio run init.\n");

    ot_vb_cfg vb_cfg;
    td_u32 supplement_config = OT_VB_SUPPLEMENT_BNR_MOT_MASK | OT_VB_SUPPLEMENT_MOTION_DATA_MASK;

    sample_comm_sys_get_default_vb_cfg(&g_vb_param, &vb_cfg);
    if (sample_comm_sys_init_with_vb_supplement(&vb_cfg, supplement_config) != TD_SUCCESS) {
        return TD_FAILURE;
    }

#ifdef SAMPLE_MEM_SHARE_ENABLE
    sample_vio_init_mem_share();
#endif

    if (sample_comm_vi_set_vi_vpss_mode(g_vio_sys_cfg.mode_type, OT_VI_AIISP_MODE_DEFAULT) != TD_SUCCESS) {
        goto sys_exit;
    }

    if (ss_mpi_sys_set_3dnr_pos(g_vio_sys_cfg.nr_pos) != TD_SUCCESS) {
        goto sys_exit;
    }

    if (sample_vio_fmu_wrap_init(&g_vio_sys_cfg, &g_vb_param.vb_size) != TD_SUCCESS) {
        goto sys_exit;
    }

    return TD_SUCCESS;
sys_exit:
    sample_comm_sys_exit();
    return TD_FAILURE;
}

static td_s32 sample_vio_start_vpss(ot_vpss_grp grp, sample_vpss_cfg *vpss_cfg)
{
    printf("shikeDebug  vio run  start vpss %d\n",grp);

    td_s32 ret;
    sample_vpss_chn_attr vpss_chn_attr = {0};

    (td_void)memcpy_s(&vpss_chn_attr.chn_attr[0], sizeof(ot_vpss_chn_attr) * OT_VPSS_MAX_PHYS_CHN_NUM,
        vpss_cfg->chn_attr, sizeof(ot_vpss_chn_attr) * OT_VPSS_MAX_PHYS_CHN_NUM);
    if (g_vio_sys_cfg.vpss_fmu[grp] == OT_FMU_MODE_WRAP) {
        vpss_chn_attr.chn0_wrap = TD_TRUE;
    }
    (td_void)memcpy_s(vpss_chn_attr.chn_enable, sizeof(vpss_chn_attr.chn_enable),
        vpss_cfg->chn_en, sizeof(vpss_chn_attr.chn_enable));
    vpss_chn_attr.chn_array_size = OT_VPSS_MAX_PHYS_CHN_NUM;
    ret = sample_common_vpss_start(grp, &vpss_cfg->grp_attr, &vpss_chn_attr);
    if (ret != TD_SUCCESS) {
        return ret;
    }

    if (vpss_cfg->nr_attr.enable == TD_TRUE) {
        if (ss_mpi_vpss_set_grp_3dnr_attr(grp, &vpss_cfg->nr_attr) != TD_SUCCESS) {
            goto stop_vpss;
        }
    }
    /* OT_FMU_MODE_WRAP is set in sample_common_vpss_start() */
    if (g_vio_sys_cfg.vpss_fmu[grp] == OT_FMU_MODE_OFF) {
        const ot_low_delay_info low_delay_info = { TD_TRUE, 200, TD_FALSE }; /* 200: lowdelay line */
        if (ss_mpi_vpss_set_chn_low_delay(grp, 0, &low_delay_info) != TD_SUCCESS) {
            goto stop_vpss;
        }
    } else if (g_vio_sys_cfg.vpss_fmu[grp] == OT_FMU_MODE_DIRECT) {
        if (ss_mpi_vpss_set_chn_fmu_mode(grp, OT_VPSS_DIRECT_CHN, g_vio_sys_cfg.vpss_fmu[grp]) != TD_SUCCESS) {
            goto stop_vpss;
        }
        if (ss_mpi_vpss_enable_chn(grp, OT_VPSS_DIRECT_CHN) != TD_SUCCESS) {
            goto stop_vpss;
        }
    }

    if (g_vio_sys_cfg.mode_type != OT_VI_ONLINE_VPSS_ONLINE) {
        ot_gdc_param gdc_param = {0};
        gdc_param.in_size.width  = g_vb_param.vb_size.width;
        gdc_param.in_size.height = g_vb_param.vb_size.height;
        gdc_param.cell_size = OT_LUT_CELL_SIZE_16;
        if (ss_mpi_vpss_set_grp_gdc_param(grp, &gdc_param) != TD_SUCCESS) {
            goto stop_vpss;
        }
    }

    return TD_SUCCESS;
stop_vpss:
    sample_common_vpss_stop(grp, vpss_cfg->chn_en, OT_VPSS_MAX_PHYS_CHN_NUM);
    return TD_FAILURE;
}

static td_void sample_vio_stop_vpss(ot_vpss_grp grp)
{
    printf("shikeDebug  vio run  stop vpss %d\n",grp);

    td_bool chn_enable[OT_VPSS_MAX_PHYS_CHN_NUM] = {TD_TRUE, TD_FALSE, TD_FALSE, TD_FALSE};

    sample_common_vpss_stop(grp, chn_enable, OT_VPSS_MAX_PHYS_CHN_NUM);
}

static td_s32 sample_vio_start_venc(ot_venc_chn venc_chn[], size_t size, td_u32 chn_num)
{
    printf("shikeDebug  vio run  start venc\n");

    td_s32 i;
    td_s32 ret;

    if (chn_num > size) {
        return TD_FAILURE;
    }

    sample_comm_vi_get_size_by_sns_type(FPGA_BT1120_14BIT, &g_venc_chn_param.venc_size);
    for (i = 0; i < (td_s32)chn_num; i++) {
        ret = sample_comm_venc_start(venc_chn[i], &g_venc_chn_param);
        if (ret != TD_SUCCESS) {
            goto exit;
        }
    }

    ret = sample_comm_venc_start_get_stream(venc_chn, chn_num);
    if (ret != TD_SUCCESS) {
        goto exit;
    }

    return TD_SUCCESS;

exit:
    for (i = i - 1; i >= 0; i--) {
        sample_comm_venc_stop(venc_chn[i]);
    }
    return TD_FAILURE;
}

static td_void sample_vio_stop_venc(ot_venc_chn venc_chn[], size_t size, td_u32 chn_num)
{
    printf("shikeDebug  vio run  stop venc\n");

    td_u32 i;

    if (chn_num > size) {
        return;
    }

    sample_comm_venc_stop_get_stream(chn_num);

    for (i = 0; i < chn_num; i++) {
        sample_comm_venc_stop(venc_chn[i]);
    }
}

static td_s32 sample_vio_start_vo(sample_vo_mode vo_mode)
{
    printf("shikeDebug  vio run  start vo\n");

    td_s32 ret;
    g_vo_cfg.vo_mode = vo_mode;

    ret = sample_comm_vo_start_vo(&g_vo_cfg);
    if (ret != TD_SUCCESS) {
        sample_print("start vo failed with 0x%x!\n", ret);
        return ret;
    }

    printf("start vo dhd%d.\n", g_vo_cfg.vo_dev);

    if ((g_vo_cfg.vo_intf_type & OT_VO_INTF_MIPI) ||
        (g_vo_cfg.vo_intf_type & OT_VO_INTF_MIPI_SLAVE)) {
        ret = sample_comm_start_mipi_tx(&g_mipi_tx_config);
        if (ret != TD_SUCCESS) {
            sample_print("start mipi tx failed with 0x%x!\n", ret);
            return ret;
        }
    }

    return TD_SUCCESS;
}

static td_void sample_vio_stop_vo(td_void)
{
    printf("shikeDebug  vio run  stop vo\n");

    if((g_vo_cfg.vo_intf_type & OT_VO_INTF_MIPI) ||
        (g_vo_cfg.vo_intf_type & OT_VO_INTF_MIPI_SLAVE)) {
        sample_comm_stop_mipi_tx(g_vo_cfg.vo_intf_type);
    }

    sample_comm_vo_stop_vo(&g_vo_cfg);
}

static td_s32 sample_vio_start_venc_and_vo(ot_vpss_grp vpss_grp[], size_t size, td_u32 grp_num)
{
    printf("shikeDebug  vio run  start venc vo\n");

    td_u32 i;
    td_s32 ret;
    sample_vo_mode vo_mode = VO_MODE_1MUX;
    const ot_vo_layer vo_layer = 0;
    ot_vo_chn vo_chn[4] = {0, 1, 2, 3};     /* 4: max chn num, 0/1/2/3 chn id */
    ot_venc_chn venc_chn[4] = {0, 1, 2, 3}; /* 4: max chn num, 0/1/2/3 chn id */

    if (grp_num > size) {
        return TD_FAILURE;
    }

    if (grp_num > 1) {
        vo_mode = VO_MODE_4MUX;
    }

    ret = sample_vio_start_venc(venc_chn, sizeof(venc_chn) / sizeof(venc_chn[0]), grp_num);
    if (ret != TD_SUCCESS) {
        goto start_venc_failed;
    }
    for (i = 0; i < grp_num; i++) {
        if (g_vio_sys_cfg.vpss_fmu[i] == OT_FMU_MODE_DIRECT) {
            sample_comm_vpss_bind_venc(vpss_grp[i], OT_VPSS_DIRECT_CHN, venc_chn[i]);
            printf("shikeDebug vpss bin venc(1) %d %d %d\n" ,vpss_grp[i], OT_VPSS_DIRECT_CHN, venc_chn[i]);
            
        } else {
            //shikeDebug
            //sample_comm_vpss_bind_venc(vpss_grp[i], OT_VPSS_CHN0, venc_chn[i]);
            //printf("shikeDebug vpss bin venc(2) %d %d %d\n" ,vpss_grp[i], OT_VPSS_CHN0, venc_chn[i]);
            sample_comm_vpss_bind_venc(vpss_grp[i], OT_VPSS_CHN0, venc_chn[i]);
            printf("shikeDebug vpss bin venc(2) %d %d %d\n" ,vpss_grp[i], OT_VPSS_CHN0, venc_chn[i]);
        }
    }

    ret = sample_vio_start_vo(vo_mode);
    if (ret != TD_SUCCESS) {
        goto start_vo_failed;
    }
    for (i = 0; i < grp_num; i++) {
        if (g_vio_sys_cfg.vpss_fmu[i] == OT_FMU_MODE_WRAP) {
            sample_comm_vpss_bind_vo(vpss_grp[i], OT_VPSS_CHN1, vo_layer, vo_chn[i]);
        } else {
            sample_comm_vpss_bind_vo(vpss_grp[i], OT_VPSS_CHN0, vo_layer, vo_chn[i]);
        }
    }

    return TD_SUCCESS;

start_vo_failed:
    for (i = 0; i < grp_num; i++) {
        if (g_vio_sys_cfg.vpss_fmu[i] == OT_FMU_MODE_DIRECT) {
            sample_comm_vpss_un_bind_venc(vpss_grp[i], OT_VPSS_DIRECT_CHN, venc_chn[i]);
        } else {
            sample_comm_vpss_un_bind_venc(vpss_grp[i], OT_VPSS_CHN0, venc_chn[i]);
        }
    }
    sample_vio_stop_venc(venc_chn, sizeof(venc_chn) / sizeof(venc_chn[0]), grp_num);
start_venc_failed:
    return TD_FAILURE;
}

static td_void sample_vio_stop_venc_and_vo(ot_vpss_grp vpss_grp[], size_t size, td_u32 grp_num)
{
    printf("shikeDebug  vio run  stop venc vo\n");

    td_u32 i;
    const ot_vo_layer vo_layer = 0;
    ot_vo_chn vo_chn[4] = {0, 1, 2, 3};     /* 4: max chn num, 0/1/2/3 chn id */
    ot_venc_chn venc_chn[4] = {0, 1, 2, 3}; /* 4: max chn num, 0/1/2/3 chn id */

    if (grp_num > size) {
        return;
    }

    for (i = 0; i < grp_num; i++) {
        if (g_vio_sys_cfg.vpss_fmu[i] == OT_FMU_MODE_WRAP) {
            sample_comm_vpss_un_bind_vo(vpss_grp[i], OT_VPSS_CHN1, vo_layer, vo_chn[i]);
        } else {
            sample_comm_vpss_un_bind_vo(vpss_grp[i], OT_VPSS_CHN0, vo_layer, vo_chn[i]);
        }
        if (g_vio_sys_cfg.vpss_fmu[i] == OT_FMU_MODE_DIRECT) {
            sample_comm_vpss_un_bind_venc(vpss_grp[i], OT_VPSS_DIRECT_CHN, venc_chn[i]);
        } else {
            sample_comm_vpss_un_bind_venc(vpss_grp[i], OT_VPSS_CHN0, venc_chn[i]);
        }
    }

    sample_vio_stop_venc(venc_chn, sizeof(venc_chn) / sizeof(venc_chn[0]), grp_num);
    sample_vio_stop_vo();
}

static td_s32 sample_vio_start_route(sample_vi_cfg *vi_cfg, sample_vpss_cfg *vpss_cfg, td_s32 route_num)
{
    printf("shikeDebug  vio run  start route have grp\n");

    td_s32 i, j, ret;
    ot_vpss_grp vpss_grp[SAMPLE_VIO_MAX_ROUTE_NUM] = {0, 1, 2, 3};

    sample_comm_vi_get_size_by_sns_type(FPGA_BT1120_14BIT, &g_vb_param.vb_size);
    if (sample_vio_sys_init() != TD_SUCCESS) {
        return TD_FAILURE;
    }

    for (i = 0; i < route_num; i++) {
        ret = sample_comm_vi_start_vi(&vi_cfg[i]);
            if (ret != TD_SUCCESS) {
            goto start_vi_failed;
        }
    }

    sample_comm_vi_bind_vpss(3, 0, 0, 0);
    printf("shikeDebug vi bin vpss(1) 3 0 0 0 \n");
    // for (i = 0; i < route_num; i++) {
    //     sample_comm_vi_bind_vpss(i, 0, vpss_grp[i], 0);
    //     printf("shikeDebug vi bin vpss(1) %d 0 %d 0\n" ,i ,vpss_grp[i]);
    // }

    for (i = 0; i < route_num; i++) {
        ret = sample_vio_start_vpss(vpss_grp[i], vpss_cfg);
        if (ret != TD_SUCCESS) {
            goto start_vpss_failed;
        }
    }

    ret = sample_vio_start_venc_and_vo(vpss_grp, SAMPLE_VIO_MAX_ROUTE_NUM, route_num);
    if (ret != TD_SUCCESS) {
        goto start_venc_and_vo_failed;
    }

    return TD_SUCCESS;

start_venc_and_vo_failed:
start_vpss_failed:
    for (j = i - 1; j >= 0; j--) {
        sample_vio_stop_vpss(vpss_grp[j]);
    }
    for (i = 0; i < route_num; i++) {
        sample_comm_vi_un_bind_vpss(i, 0, vpss_grp[i], 0);
    }
start_vi_failed:
    for (j = i - 1; j >= 0; j--) {
        sample_comm_vi_stop_vi(&vi_cfg[j]);
    }
    sample_comm_sys_exit();
    return TD_FAILURE;
}

static td_void sample_vio_stop_route(sample_vi_cfg *vi_cfg, td_s32 route_num)
{
    printf("shikeDebug  vio run  stop route\n");

    td_s32 i;
    ot_vpss_grp vpss_grp[SAMPLE_VIO_MAX_ROUTE_NUM] = {0, 1, 2, 3};

    sample_vio_stop_venc_and_vo(vpss_grp, SAMPLE_VIO_MAX_ROUTE_NUM, route_num);
    for (i = 0; i < route_num; i++) {
        sample_vio_stop_vpss(vpss_grp[i]);
    }
    sample_comm_vi_un_bind_vpss(3, 0, 0, 0);
    for (i = 0; i < route_num; i++) {
        sample_comm_vi_stop_vi(&vi_cfg[i]);
    }

    sample_comm_sys_exit();
}

static td_s32 sample_vio_all_mode(td_void)
{
    printf("shikeDebug  vio run all mode.\n");

    sample_vi_cfg vi_cfg[1];
    sample_vpss_cfg vpss_cfg;
    sample_sns_type sns_type = FPGA_BT1120_14BIT;

    g_vio_sys_cfg.mode_type = OT_VI_OFFLINE_VPSS_OFFLINE;
    g_vio_sys_cfg.vi_fmu[0] = OT_FMU_MODE_OFF;
    g_vb_param.blk_num[0] =  6; /* raw_vb num 6 or 3 */

    sample_comm_vi_get_vi_cfg_by_fmu_mode(sns_type, g_vio_sys_cfg.vi_fmu[0], &vi_cfg[0]);
    sample_comm_vpss_get_default_vpss_cfg(&vpss_cfg, g_vio_sys_cfg.vpss_fmu[0]);

    if (sample_vio_start_route(vi_cfg, &vpss_cfg, g_vio_sys_cfg.route_num) != TD_SUCCESS) {
        return TD_FAILURE;
    }

    // sample_get_char();

    // sample_vio_stop_route(vi_cfg, g_vio_sys_cfg.route_num);

    return TD_SUCCESS;
}

static td_s32 sample_vio_all_mode_stop(td_void)
{
    printf("shikeDebug  vio run all mode.\n");

    sample_vi_cfg vi_cfg[1];
    sample_vpss_cfg vpss_cfg;
    sample_sns_type sns_type = FPGA_BT1120_14BIT;

    g_vio_sys_cfg.mode_type = OT_VI_OFFLINE_VPSS_OFFLINE;
    g_vio_sys_cfg.vi_fmu[0] = OT_FMU_MODE_OFF;
    g_vb_param.blk_num[0] =  6; /* raw_vb num 6 or 3 */

    sample_comm_vi_get_vi_cfg_by_fmu_mode(sns_type, g_vio_sys_cfg.vi_fmu[0], &vi_cfg[0]);
    sample_comm_vpss_get_default_vpss_cfg(&vpss_cfg, g_vio_sys_cfg.vpss_fmu[0]);

    // if (sample_vio_start_route(vi_cfg, &vpss_cfg, g_vio_sys_cfg.route_num) != TD_SUCCESS) {
    //     return TD_FAILURE;
    // }

    // sample_get_char();

    sample_vio_stop_route(vi_cfg, g_vio_sys_cfg.route_num);

    return TD_SUCCESS;
}



static td_void sample_vio_handle_sig(td_s32 signo)
{
    if (signo == SIGINT || signo == SIGTERM) {
        g_sig_flag = 1;
    }
}

static td_void sample_register_sig_handler(td_void (*sig_handle)(td_s32))
{
    struct sigaction sa;

    (td_void)memset_s(&sa, sizeof(struct sigaction), 0, sizeof(struct sigaction));
    sa.sa_handler = sig_handle;
    sa.sa_flags = 0;
    sigaction(SIGINT, &sa, TD_NULL);
    sigaction(SIGTERM, &sa, TD_NULL);
}

static td_s32 sample_vio_msg_proc_vb_pool_share(td_s32 pid)
{
    td_s32 ret;
    td_u32 i;
    td_bool isp_states[OT_VI_MAX_PIPE_NUM];
#ifndef SAMPLE_MEM_SHARE_ENABLE
    ot_vb_common_pools_id pools_id = {0};

    if (ss_mpi_vb_get_common_pool_id(&pools_id) != TD_SUCCESS) {
        sample_print("get common pool_id failed!\n");
        return TD_FAILURE;
    }

    for (i = 0; i < pools_id.pool_cnt; ++i) {
        if (ss_mpi_vb_pool_share(pools_id.pool[i], pid) != TD_SUCCESS) {
            sample_print("vb pool share failed!\n");
            return TD_FAILURE;
        }
    }
#endif
    ret = sample_comm_vi_get_isp_run_state(isp_states, OT_VI_MAX_PIPE_NUM);
    if (ret != TD_SUCCESS) {
        sample_print("get isp states fail\n");
        return TD_FAILURE;
    }

    for (i = 0; i < OT_VI_MAX_PIPE_NUM; i++) {
        if (!isp_states[i]) {
            continue;
        }
        ret = ss_mpi_isp_mem_share(i, pid);
        if (ret != TD_SUCCESS) {
            sample_print("ss_mpi_isp_mem_share vi_pipe %u, pid %d fail\n", i, pid);
        }
    }

    return TD_SUCCESS;
}

static td_void sample_vio_msg_proc_vb_pool_unshare(td_s32 pid)
{
    td_s32 ret;
    td_u32 i;
    td_bool isp_states[OT_VI_MAX_PIPE_NUM];
#ifndef SAMPLE_MEM_SHARE_ENABLE
    ot_vb_common_pools_id pools_id = {0};
    if (ss_mpi_vb_get_common_pool_id(&pools_id) == TD_SUCCESS) {
        for (i = 0; i < pools_id.pool_cnt; ++i) {
            ret = ss_mpi_vb_pool_unshare(pools_id.pool[i], pid);
            if (ret != TD_SUCCESS) {
                sample_print("ss_mpi_vb_pool_unshare vi_pipe %u, pid %d fail\n", pools_id.pool[i], pid);
            }
        }
    }
#endif
    ret = sample_comm_vi_get_isp_run_state(isp_states, OT_VI_MAX_PIPE_NUM);
    if (ret != TD_SUCCESS) {
        sample_print("get isp states fail\n");
        return;
    }

    for (i = 0; i < OT_VI_MAX_PIPE_NUM; i++) {
        if (!isp_states[i]) {
            continue;
        }
        ret = ss_mpi_isp_mem_unshare(i, pid);
        if (ret != TD_SUCCESS) {
            sample_print("ss_mpi_isp_mem_unshare vi_pipe %u, pid %d fail\n", i, pid);
        }
    }
}

static td_s32 sample_vio_ipc_msg_proc(const sample_ipc_msg_req_buf *msg_req_buf,
    td_bool *is_need_fb, sample_ipc_msg_res_buf *msg_res_buf)
{
    td_s32 ret;

    if (msg_req_buf == TD_NULL || is_need_fb == TD_NULL) {
        return TD_FAILURE;
    }

    /* need feedback default */
    *is_need_fb = TD_TRUE;

    switch ((sample_msg_type)msg_req_buf->msg_type) {
        case SAMPLE_MSG_TYPE_VB_POOL_SHARE_REQ: {
            if (msg_res_buf == TD_NULL) {
                return TD_FAILURE;
            }
            ret = sample_vio_msg_proc_vb_pool_share(msg_req_buf->msg_data.pid);
            msg_res_buf->msg_type = SAMPLE_MSG_TYPE_VB_POOL_SHARE_RES;
            msg_res_buf->msg_data.is_req_success = (ret == TD_SUCCESS) ? TD_TRUE : TD_FALSE;
            break;
        }
        case SAMPLE_MSG_TYPE_VB_POOL_UNSHARE_REQ: {
            if (msg_res_buf == TD_NULL) {
                return TD_FAILURE;
            }
            sample_vio_msg_proc_vb_pool_unshare(msg_req_buf->msg_data.pid);
            msg_res_buf->msg_type = SAMPLE_MSG_TYPE_VB_POOL_UNSHARE_RES;
            msg_res_buf->msg_data.is_req_success = TD_TRUE;
            break;
        }
        default: {
            printf("unsupported msg type(%ld)!\n", msg_req_buf->msg_type);
            return TD_FAILURE;
        }
    }
    return TD_SUCCESS;
}

void app_start()
{
    sample_register_sig_handler(sample_vio_handle_sig);

    if (sample_ipc_server_init(sample_vio_ipc_msg_proc) != TD_SUCCESS) {
        printf("sample_ipc_server_init failed!!!\n");
    }
    
    //ret = sample_vio_execute_case(index);
    td_s32 ret = sample_vio_all_mode();
    if ((ret == TD_SUCCESS) && (g_sig_flag == 0)) {
        printf("\033[0;32mprogram exit normally!\033[0;39m\n");
    } else {
        printf("\033[0;31mprogram exit abnormally!\033[0;39m\n");
    }

}

// 全局互斥锁
pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

int Open_Infrared_Vio(){
    pthread_mutex_lock(&g_mutex);
    printf("open infrared vio ...\n");
    //TODO
    app_start();
    usleep(1000);
    pthread_mutex_unlock(&g_mutex);

    printf("open infrared done\n");
    return 0;
}

int Close_Infrared_Vio(){
    pthread_mutex_lock(&g_mutex);
    printf("close infrared vio ...\n");
    
    sample_vio_all_mode_stop();

    usleep(100*1000);
    sample_ipc_server_deinit();
    usleep(1000);
    pthread_mutex_unlock(&g_mutex);
    printf("close infrared vio done\n");
    return 0;
}

int Open_Colorcamera_Vio(){
    pthread_mutex_lock(&g_mutex);
    printf("open colorcamera vio ...\n");
    //TODO
    usleep(1000);
    pthread_mutex_unlock(&g_mutex);
    printf("open colorcamera vio done\n");
    return 0;
}

int Close_Colorcamera_Vio(){
    pthread_mutex_lock(&g_mutex);
    printf("close colorcamera vio ...\n");
    //TODO
    usleep(1000);
    pthread_mutex_unlock(&g_mutex);
    printf("close colorcamera vio done\n");
    return 0;
}

// 定义系统状态
typedef enum {
    STATE_NO,
    STATE_A,  // 执行函数A
    STATE_B   // 执行函数B
} SystemState;

// 共享资源结构体
typedef struct {
    pthread_mutex_t mutex;      // 互斥锁
    pthread_cond_t cond;        // 条件变量
    int button_id;              // 按钮编号
    SystemState state;          // 当前系统状态
    int exit_flag;              // 退出标志
} ButtonEvent;

ButtonEvent g_event;            // 全局事件对象

// 子线程函数：监控按键
void* button_monitor_thread(void* arg) {
    int fd = open("/dev/buttons", O_RDONLY);
    if (fd < 0) {
        perror("Failed to open buttons device");
        return NULL;
    }

    unsigned char button_state;
    while (1) {
        // 读取按键状态（阻塞模式）
        if (read(fd, &button_state, 1) != 1) {
            perror("Read button state failed");
            break;
        }

        // 检查按钮编号是否非0
        if (button_state != 0) {
            // 加锁保护共享资源
            pthread_mutex_lock(&g_event.mutex);
            
            // 更新按钮编号
            g_event.button_id = button_state;
            
            // 通知主线程
            pthread_cond_signal(&g_event.cond);
            
            // 解锁
            pthread_mutex_unlock(&g_event.mutex);
        }

        // 检查退出标志
        pthread_mutex_lock(&g_event.mutex);
        if (g_event.exit_flag) {
            pthread_mutex_unlock(&g_event.mutex);
            break;
        }
        pthread_mutex_unlock(&g_event.mutex);
        usleep(5*1000);
    }

    close(fd);
    return NULL;
}

int main() {
    // 初始化互斥锁和条件变量
    pthread_mutex_init(&g_event.mutex, NULL);
    pthread_cond_init(&g_event.cond, NULL);
    g_event.button_id = 0;
    g_event.state = STATE_A;    // 默认执行函数A
    g_event.exit_flag = 0;

    Open_Infrared_Vio();

    // 创建子线程
    pthread_t monitor_thread;
    if (pthread_create(&monitor_thread, NULL, button_monitor_thread, NULL) != 0) {
        perror("Failed to create monitor thread");
        return -1;
    }

    // 主线程：处理按钮事件
    while (1) {
        // 加锁
        pthread_mutex_lock(&g_event.mutex);
        
        // 等待按钮事件（按钮编号非0）
        while (g_event.button_id == 0 && !g_event.exit_flag) {
            pthread_cond_wait(&g_event.cond, &g_event.mutex);
        }
        
        // 检查是否需要退出
        if (g_event.exit_flag) {
            pthread_mutex_unlock(&g_event.mutex);
            break;
        }
        
        // 处理按钮事件
        printf("Main thread: Button %d pressed!\n", g_event.button_id);
        
        // 重置按钮编号
        g_event.button_id = 0;
        
        // 解锁
        pthread_mutex_unlock(&g_event.mutex);
        
        // 执行按钮对应的操作
        // 根据当前状态执行相应函数
        if (g_event.state == STATE_A) {
            // 释放互斥锁，允许按钮线程在执行函数A时更新状态
            g_event.state = STATE_B;
            Close_Infrared_Vio();
            Open_Colorcamera_Vio();
            pthread_mutex_unlock(&g_event.mutex);
        } else if (g_event.state == STATE_B) {
            g_event.state = STATE_A;
            Close_Colorcamera_Vio();
            Open_Infrared_Vio();
            pthread_mutex_unlock(&g_event.mutex);
        } else {
            pthread_mutex_unlock(&g_event.mutex);
        }
        usleep(1000);
    }

    // 等待子线程退出
    g_event.exit_flag = 1;
    pthread_join(monitor_thread, NULL);

    // 清理资源
    pthread_mutex_destroy(&g_event.mutex);
    pthread_cond_destroy(&g_event.cond);

    return 0;
}
