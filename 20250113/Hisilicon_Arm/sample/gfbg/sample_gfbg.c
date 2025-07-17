/*
  Copyright (c), 2001-2024, Shenshu Tech. Co., Ltd.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <math.h>
#include <unistd.h>
#include <signal.h>
#include <limits.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <linux/fb.h>

#include "securec.h"
#include "loadbmp.h"

#include "gfbg.h"
#include "ot_common_tde.h"
#include "ss_mpi_tde.h"
#include "sample_comm.h"

#define FILE_LENGTH_MAX 12
#define CMAP_LENGTH_MAX 256
#define WIDTH_1080P 1920
#define HEIGHT_1080P 1080
#define WIDTH_800 800
#define HEIGHT_600 600

#define SAMPLE_IMAGE_WIDTH     300
#define SAMPLE_IMAGE_HEIGHT    150
#define SAMPLE_IMAGE_NUM       20
#define GFBG_RED_1555          0xFC00
#define GFBG_RED_8888          0xFFff0000
#define WIDTH_1920             1920
#define HEIGHT_1080            1080

#define GRAPHICS_LAYER_G0      0
#define GRAPHICS_LAYER_G1      1
#define GRAPHICS_LAYER_G2      2
#define GRAPHICS_LAYER_G3      3

#define SAMPLE_IMAGE1_PATH        "./res/%u.bmp"
#define SAMPLE_IMAGE2_PATH        "./res/1280_720.bits"
#define SAMPLE_CURSOR_PATH        "./res/cursor.bmp"

//#if (defined(CONFIG_OT_GFBG_SUPPORT) && defined(CONFIG_OT_VO_SUPPORT))
//#define GFBG_BE_WITH_VO    1
//#else
#define GFBG_BE_WITH_VO    0
//#endif

#define GFBG_INPUT_PARAMETERS_NUM 2

static td_char g_exit_flag = 0;
static int g_sample_gfbg_exit = 0;
ot_vo_intf_type g_vo_intf_type = OT_VO_INTF_BT1120;
ot_vo_dev g_vo_dev = 0;
osd_color_format g_osd_color_fmt = OSD_COLOR_FORMAT_RGB1555;

static struct fb_bitfield g_r16 = {10, 5, 0};
static struct fb_bitfield g_g16 = {5, 5, 0};
static struct fb_bitfield g_b16 = {0, 5, 0};
static struct fb_bitfield g_a16 = {15, 1, 0};

static struct fb_bitfield g_a32 = {24, 8, 0};
static struct fb_bitfield g_r32 = {16, 8, 0};
static struct fb_bitfield g_g32 = {8,  8, 0};
static struct fb_bitfield g_b32 = {0,  8, 0};

static struct fb_bitfield g_a4 = {0, 0, 0};
static struct fb_bitfield g_r4 = {0, 4, 0};
static struct fb_bitfield g_g4 = {0, 4, 0};
static struct fb_bitfield g_b4 = {0, 4, 0};

td_u16 g_cmap_red[CMAP_LENGTH_MAX] = {0xff, 0, 0, 0xff};
td_u16 g_cmap_green[CMAP_LENGTH_MAX] = {0, 0xff, 0, 0xff};
td_u16 g_cmap_blue[CMAP_LENGTH_MAX] = {0, 0, 0xff, 0xff};
td_u16 g_cmap_alpha[CMAP_LENGTH_MAX] = {0xff, 0xff, 0xff, 0xff};

pthread_t g_gfbg_thread = 0;
pthread_t g_gfbg_thread1 = 0;

td_phys_addr_t g_phyaddr = 0;
td_phys_addr_t g_canvas_addr = 0;

#ifdef CONFIG_SUPPORT_SAMPLE_ROTATION
static td_phys_addr_t g_canvas_phy = 0;
static td_void *g_canvas_vir = NULL;
static td_phys_addr_t g_picture_phy = 0;
static td_void *g_picture_vir = NULL;
#endif

typedef struct {
    td_s32 fd; /* fb's file describe */
    td_s32 layer; /* which graphic layer */
    td_s32 ctrlkey; /* {0,1,2,3}={1buffer, 2buffer, 0buffer pan display, 0buffer refresh} */
    td_bool compress; /* image compressed or not */
    ot_fb_color_format color_format; /* color format. */
} pthread_gfbg_sample_info;

typedef struct {
    ot_vo_dev vo_dev;
    ot_vo_intf_type ot_vo_intf_type;
}vo_device_info;

#ifdef CONFIG_SUPPORT_SAMPLE_ROTATION
static td_s32 gfbg_put_layer_info(pthread_gfbg_sample_info *info);
static td_s32 gfbg_get_canvas(pthread_gfbg_sample_info *info);
static td_void gfbg_put_canvas(td_void);
static td_void gfbg_init_surface(ot_tde_surface *, ot_tde_surface *, ot_tde_rect *, ot_tde_rect *, td_u32);
static td_s32 gfbg_draw(pthread_gfbg_sample_info *info);
static td_s32 gfbg_refresh(const pthread_gfbg_sample_info *info);
static td_void gfbg_get_var_by_format(pthread_gfbg_sample_info *info, struct fb_var_screeninfo *var_info);
static td_void gfbg_put_rotation_degree(struct fb_var_screeninfo *var_info, ot_fb_rotate_mode *rotate_mode);
static td_s32 gfbg_put_rotation(pthread_gfbg_sample_info *info);
static td_void gfbg_rotate(ot_vo_dev vo_dev);
#endif
static td_void sample_gfbg_to_exit_signal(td_void);

static int sample_gfbg_getchar(td_void)
{
    int c;
    if (g_sample_gfbg_exit == 1) {
        sample_gfbg_to_exit_signal();
        //sample_comm_sys_exit();
        printf("\033[0;31mprogram exit abnormally!\033[0;39m\n");
        return TD_FAILURE;
    }

    c = getchar();

    if (g_sample_gfbg_exit == 1) {
        sample_gfbg_to_exit_signal();
        //sample_comm_sys_exit();
        printf("\033[0;31mprogram exit abnormally!\033[0;39m\n");
        return TD_FAILURE;
    }

    return c;
}

static td_s32 sample_gfbg_load_bmp(const char *filename, td_u8 *addr)
{
    osd_surface surface;
    osd_bit_map_file_header bmp_file_header;
    osd_bit_map_info bmp_info;

    if (get_bmp_info(filename, &bmp_file_header, &bmp_info) < 0) {
        sample_print("get_bmp_info err!\n");
        return TD_FAILURE;
    }
    surface.color_format = g_osd_color_fmt;
    create_surface_by_bit_map(filename, &surface, addr);
    return TD_SUCCESS;
}

static td_void sample_gfbg_to_exit(td_void)
{
    td_char ch;
    while (1) {
        printf("\npress 'q' to exit this sample.\n");
        while ((ch = (char)sample_gfbg_getchar()) == '\n') {};
        if (ch == 'q') {
            g_exit_flag = ch;
            break;
        } else {
            printf("input invalid! please try again.\n");
            //shikeDebug
            break;
        }
    }
    if (g_gfbg_thread != 0) {
        pthread_join(g_gfbg_thread, 0);
        g_gfbg_thread = 0;
    }

    if (g_gfbg_thread1 != 0) {
        pthread_join(g_gfbg_thread1, 0);
        g_gfbg_thread1 = 0;
    }
    return;
}

static td_s32 sample_get_file_name(pthread_gfbg_sample_info *info, td_char *file, td_u32 file_length)
{
    switch (info->layer) {
        case GRAPHICS_LAYER_G0:
            if (strncpy_s(file, file_length, "/dev/fb0", strlen("/dev/fb0")) != EOK) {
                printf("%s:%d:strncpy_s failed.\n", __FUNCTION__, __LINE__);
                return TD_FAILURE;
            }
            break;
        case GRAPHICS_LAYER_G1:
            if (strncpy_s(file, file_length, "/dev/fb1", strlen("/dev/fb1")) != EOK) {
                printf("%s:%d:strncpy_s failed.\n", __FUNCTION__, __LINE__);
                return TD_FAILURE;
            }
            break;
        case GRAPHICS_LAYER_G2:
            if (strncpy_s(file, file_length, "/dev/fb2", strlen("/dev/fb2")) != EOK) {
                printf("%s:%d:strncpy_s failed.\n", __FUNCTION__, __LINE__);
                return TD_FAILURE;
            }
            break;
        case GRAPHICS_LAYER_G3:
            if (strncpy_s(file, file_length, "/dev/fb2", strlen("/dev/fb2")) != EOK) {
                printf("%s:%d:strncpy_s failed.\n", __FUNCTION__, __LINE__);
                return TD_FAILURE;
            }
            break;
        default:
            if (strncpy_s(file, file_length, "/dev/fb0", strlen("/dev/fb0")) != EOK) {
                printf("%s:%d:strncpy_s failed.\n", __FUNCTION__, __LINE__);
                return TD_FAILURE;
            }
            break;
    }
    return TD_SUCCESS;
}

static td_s32 sample_init_frame_buffer(pthread_gfbg_sample_info *info, const char *input_file)
{
    td_bool show;
    ot_fb_point point = {0, 0};
    td_char file[PATH_MAX + 1] = {0};

    if (strlen(input_file) > PATH_MAX || realpath(input_file, file) == TD_NULL) {
        return TD_FAILURE;
    }
    /* step 1. open framebuffer device overlay 0 */
    info->fd = open(file, O_RDWR, 0);
    if (info->fd < 0) {
        sample_print("open %s failed!\n", file);
        return TD_FAILURE;
    }

    show = TD_FALSE;
    if (ioctl(info->fd, FBIOPUT_SHOW_GFBG, &show) < 0) {
        sample_print("FBIOPUT_SHOW_GFBG failed!\n");
        close(info->fd);
        info->fd = -1;
        return TD_FAILURE;
    }

    /* step 2. set the screen original position */
    switch (info->ctrlkey) {
        case 3: /* 3 mouse case */
            point.x_pos = 150; /* 150 x pos */
            point.y_pos = 150; /* 150 y pos */
            break;
        default:
            point.x_pos = 0;
            point.y_pos = 0;
    }

    if (ioctl(info->fd, FBIOPUT_SCREEN_ORIGIN_GFBG, &point) < 0) {
        sample_print("set screen original show position failed!\n");
        close(info->fd);
        info->fd = -1;
        return TD_FAILURE;
    }

    return TD_SUCCESS;
}

static td_s32 sample_init_var(pthread_gfbg_sample_info *info)
{
    struct fb_var_screeninfo var;

    if (ioctl(info->fd, FBIOGET_VSCREENINFO, &var) < 0) {
        sample_print("get variable screen info failed!\n");
        return TD_FAILURE;
    }

    switch (info->color_format) {
        case OT_FB_FORMAT_ARGB8888:
            var.transp = g_a32;
            var.red = g_r32;
            var.green = g_g32;
            var.blue = g_b32;
            var.bits_per_pixel = 32; /* 32 for 4 byte */
            g_osd_color_fmt = OSD_COLOR_FORMAT_RGB8888;
            break;
        default:
            var.transp = g_a16;
            var.red = g_r16;
            var.green = g_g16;
            var.blue = g_b16;
            var.bits_per_pixel = 16; /* 16 for 2 byte */
            break;
    }

    switch (info->ctrlkey) {
        case 3: /* 3 mouse case */
            var.xres_virtual = 48; /* 48 for alg data */
            var.yres_virtual = 48; /* 48 for alg data */
            var.xres = 48; /* 48 for alg data */
            var.yres = 48; /* 48 for alg data */
            break;
        default:
            var.xres_virtual = WIDTH_1080P;
            var.yres_virtual = HEIGHT_1080P * 2; /* 2 for 2buf */
            var.xres = WIDTH_1080P;
            var.yres = HEIGHT_1080P;
    }
    var.activate       = FB_ACTIVATE_NOW;

    if (ioctl(info->fd, FBIOPUT_VSCREENINFO, &var) < 0) {
        sample_print("put variable screen info failed!\n");
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_put_alpha_and_colorkey(pthread_gfbg_sample_info *info, ot_fb_color_format clr_fmt)
{
    td_s32 ret;
    ot_fb_alpha alpha = {0};
    ot_fb_colorkey color_key;
    sample_print("expected: the red box will appear!\n");
    sleep(2); /* 2 second */
    alpha.pixel_alpha = TD_TRUE;
    alpha.alpha0 = 0x0;
    alpha.alpha1 = 0x0;
    if (ioctl(info->fd, FBIOPUT_ALPHA_GFBG,  &alpha) < 0) {
        sample_print("set alpha failed!\n");
        return TD_FAILURE;
    }
    sample_print("expected: after set alpha = 0, the red box will disappear!\n");
    sleep(2); /* 2 second */

    alpha.alpha0 = 0;
    alpha.alpha1 = 0xFF;
    if (ioctl(info->fd, FBIOPUT_ALPHA_GFBG,  &alpha) < 0) {
        sample_print("set alpha failed!\n");
        return TD_FAILURE;
    }
    sample_print("expected:after set set alpha = 0xFF, the red box will appear again!\n");
    sleep(2); /* 2 second */

    sample_print("expected: the red box will erased by colorkey!\n");
    color_key.enable = TD_TRUE;
    color_key.value = (clr_fmt == OT_FB_FORMAT_ARGB8888) ? GFBG_RED_8888 : GFBG_RED_1555;
    ret = ioctl(info->fd, FBIOPUT_COLORKEY_GFBG, &color_key);
    if (ret < 0) {
        sample_print("FBIOPUT_COLORKEY_GFBG failed!\n");
        return TD_FAILURE;
    }
    sleep(2); /* 2 second */
    sample_print("expected: the red box will appear again!\n");
    color_key.enable = TD_FALSE;
    ret = ioctl(info->fd, FBIOPUT_COLORKEY_GFBG, &color_key);
    if (ret < 0) {
        sample_print("FBIOPUT_COLORKEY_GFBG failed!\n");
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_void sample_draw_rect_by_cpu(td_void *ptemp, ot_fb_color_format clr_fmt, struct fb_var_screeninfo *var)
{
    td_s32 x, y;
    for (y = 100; y < 300; y++) { /* 100 300 for y */
        for (x = 0; x < 300; x++) { /* 300 for x */
            if (clr_fmt == OT_FB_FORMAT_ARGB8888) {
                *((td_u32*)ptemp + y * var->xres + x) = GFBG_RED_8888;
            } else {
                *((td_u16*)ptemp + y * var->xres + x) = GFBG_RED_1555;
            }
        }
    }
    return;
}

static td_s32 sample_time_to_play(pthread_gfbg_sample_info *info, td_u8 *show_screen, td_u32 fix_screen_stride)
{
    td_bool show;
    td_void *ptemp = TD_NULL;
    td_s32 i;
    struct fb_var_screeninfo var;
    ot_fb_color_format clr_fmt;

    if (ioctl(info->fd, FBIOGET_VSCREENINFO, &var) < 0) {
        sample_print("get variable screen info failed!\n");
        return TD_FAILURE;
    }

    show = TD_TRUE;
    if (ioctl(info->fd, FBIOPUT_SHOW_GFBG, &show) < 0) {
        sample_print("FBIOPUT_SHOW_GFBG failed!\n");
        return TD_FAILURE;
    }

    switch (info->color_format) {
        case OT_FB_FORMAT_ARGB8888:
            clr_fmt = OT_FB_FORMAT_ARGB8888;
            break;
        default:
            clr_fmt = OT_FB_FORMAT_ARGB1555;
            break;
    }
    /* only for G0 or G1 */
    if ((info->layer != GRAPHICS_LAYER_G0) && (info->layer != GRAPHICS_LAYER_G1)) {
        return TD_SUCCESS;
    }
    for (i = 0; i < 1; i++) {
        if (i % 2) { /* 2 for 0 or 1 */
            var.yoffset = var.yres;
        } else {
            var.yoffset = 0;
        }
        ptemp = (show_screen + var.yres * fix_screen_stride * (i % 2)); /* 2 for 0 or 1 */
        sample_draw_rect_by_cpu(ptemp, clr_fmt, &var);
        /*
         * note : not acting on ARGB8888, for ARGB8888 format image's alpha, you can change ptemp[x][y]'s value
         * GFBG_RED_8888 = 0xffff00000 means alpha=255(show),red.0x00ff0000 means alpha=0,red(hide).
         */
        if (sample_put_alpha_and_colorkey(info, clr_fmt) != TD_SUCCESS) {
            return TD_FAILURE;
        }
        sleep(2); /* 2 second */
    }
    return TD_SUCCESS;
}

static td_s32 sample_quick_copy_by_tde(td_u32 byte_per_pixel, ot_tde_color_format tde_clr_fmt,
    td_u32 fix_screen_stride, td_u64 hide_screen_phy)
{
    td_s32 handle;
    ot_tde_rect src_rect = {0};
    ot_tde_rect dst_rect = {0};
    ot_tde_surface src_surface = {0};
    ot_tde_surface dst_surface = {0};
    ot_tde_single_src single_src = {0};

    /* TDE job step 0. open tde */
    src_rect.pos_x = dst_rect.pos_x = 0;
    src_rect.pos_y = dst_rect.pos_y = 0;
    src_rect.height = SAMPLE_IMAGE_HEIGHT;
    src_rect.width = SAMPLE_IMAGE_WIDTH;
    dst_rect.height = src_rect.height;
    dst_rect.width = src_rect.width;

    dst_surface.color_format = tde_clr_fmt;
    dst_surface.width = WIDTH_1080P;
    dst_surface.height = HEIGHT_1080P;
    dst_surface.stride = fix_screen_stride;
    dst_surface.phys_addr = (td_phys_addr_t)hide_screen_phy;

    src_surface.color_format = tde_clr_fmt;
    src_surface.width = SAMPLE_IMAGE_WIDTH;
    src_surface.height = SAMPLE_IMAGE_HEIGHT;
    src_surface.stride = byte_per_pixel * SAMPLE_IMAGE_WIDTH;
    src_surface.phys_addr = g_phyaddr;
    src_surface.support_alpha_ex_1555 = TD_TRUE;
    src_surface.alpha_max_is_255  = TD_TRUE;
    src_surface.alpha0 = 0XFF;
    src_surface.alpha1 = 0XFF;

    /* TDE job step 1. start job */
    handle = ss_tde_begin_job();
    if (handle == OT_ERR_TDE_INVALID_HANDLE) {
        return TD_FAILURE;
    }
    single_src.src_surface = &src_surface;
    single_src.dst_surface = &dst_surface;
    single_src.src_rect = &src_rect;
    single_src.dst_rect = &dst_rect;
    if (ss_tde_quick_copy(handle, &single_src) < 0) {
        sample_print("tde_quick_copy:%d failed!\n", __LINE__);
        ss_tde_cancel_job(handle);
        return TD_FAILURE;
    }

    /* TDE job step 2. submit job */
    if (ss_tde_end_job(handle, TD_FALSE, TD_TRUE, 10) < 0) { /* 10 timeout */
        sample_print("line:%d,tde_end_job failed!\n", __LINE__);
        ss_tde_cancel_job(handle);
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_void sample_get_param(pthread_gfbg_sample_info *info, ot_fb_color_format *clr_fmt,
    ot_tde_color_format *tde_clr_fmt, td_u32 *color)
{
    switch (info->color_format) {
        case OT_FB_FORMAT_ARGB8888:
            *clr_fmt = OT_FB_FORMAT_ARGB8888;
            *color = GFBG_RED_8888;
            *tde_clr_fmt = OT_TDE_COLOR_FORMAT_ARGB8888;
            break;
        default:
            *clr_fmt = OT_FB_FORMAT_ARGB1555;
            *color = GFBG_RED_1555;
            *tde_clr_fmt = OT_TDE_COLOR_FORMAT_ARGB1555;
            break;
    }
    return;
}

static td_s32 sample_draw_line_by_cpu(pthread_gfbg_sample_info *info, td_u8 *show_screen, td_u32 fix_screen_stride,
    td_u64 *hide_screen_phy, td_u32 i)
{
    td_s32 x, y;
    td_u8 *hide_screen = TD_NULL;
    td_void *show_line = TD_NULL;
    struct fb_var_screeninfo var;
    struct fb_fix_screeninfo fix;
    ot_fb_color_format clr_fmt;
    ot_tde_color_format tde_clr_fmt = OT_TDE_COLOR_FORMAT_ARGB1555;
    td_u32 color = GFBG_RED_1555;

    if (ioctl(info->fd, FBIOGET_VSCREENINFO, &var) < 0) {
        sample_print("get variable screen info failed!\n");
        return TD_FAILURE;
    }

    if (ioctl(info->fd, FBIOGET_FSCREENINFO, &fix) < 0) {
        return TD_FAILURE;
    }

    sample_get_param(info, &clr_fmt, &tde_clr_fmt, &color);
    /* TDE step1: draw two red line */
    var.yoffset = (i % 2 == 1) ? var.yres : 0; /* 2 for 0 1 */

    hide_screen = show_screen + (fix_screen_stride * var.yres) * (i % 2); /* 2 for 0 1 */
    if  (memset_s(hide_screen, fix_screen_stride * var.yres, 0x00, fix_screen_stride * var.yres) != EOK) {
        return TD_FAILURE;
    }
    *hide_screen_phy = fix.smem_start + (i % 2) * fix_screen_stride * var.yres; /* 2 for 0 1 */
    show_line = hide_screen;
    for (y = (HEIGHT_1080P / 2 - 2); y < (HEIGHT_1080P / 2 + 2); y++) { /* 2 for alg data */
        for (x = 0; x < WIDTH_1080P; x++) {
            if (clr_fmt == OT_FB_FORMAT_ARGB8888) {
                *((td_u32*)show_line + y * var.xres + x) = color;
            } else {
                *((td_u16*)show_line + y * var.xres + x) = color;
            }
        }
    }
    for (y = 0; y < HEIGHT_1080P; y++) {
        for (x = (WIDTH_1080P / 2 - 2); x < (WIDTH_1080P / 2 + 2); x++) { /* 2 for alg data */
            if (clr_fmt == OT_FB_FORMAT_ARGB8888) {
                *((td_u32*)show_line + y * var.xres + x) = color;
            } else {
                *((td_u16*)show_line + y * var.xres + x) = color;
            }
        }
    }
    if (ioctl(info->fd, FBIOPAN_DISPLAY, &var) < 0) {
            sample_print("FBIOPAN_DISPLAY failed!\n");
            return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_draw_line_and_picture(pthread_gfbg_sample_info *info, td_void *viraddr, td_u8 *show_screen,
    td_u32 fix_screen_stride)
{
    td_u32 i;
    struct fb_var_screeninfo var;
    struct fb_fix_screeninfo fix;
    td_u64 hide_screen_phy = 0;
    td_u32 byte_per_pixel;
    ot_fb_color_format clr_fmt;
    ot_tde_color_format tde_clr_fmt = OT_TDE_COLOR_FORMAT_ARGB1555;
    td_u32 color = GFBG_RED_1555;
    td_char image_name[128]; /* 128 for char length */
    td_u8 *dst = TD_NULL;

    sample_get_param(info, &clr_fmt, &tde_clr_fmt, &color);

    if (ioctl(info->fd, FBIOGET_VSCREENINFO, &var) < 0) {
        return TD_FAILURE;
    }
    byte_per_pixel = var.bits_per_pixel / 8; /* 8 for 1 byte */

    if (ioctl(info->fd, FBIOGET_FSCREENINFO, &fix) < 0) {
        return TD_FAILURE;
    }

    sample_print("expected:two red line!\n");
    for (i = 0; i < SAMPLE_IMAGE_NUM; i++) {
        if (g_exit_flag == 'q') {
            printf("process exit...\n");
            break;
        }

        if (sample_draw_line_by_cpu(info, show_screen, fix_screen_stride, &hide_screen_phy, i) != TD_SUCCESS) {
            return TD_FAILURE;
        }

        /* TDE step2: draw gui picture */
        if (snprintf_s(image_name, sizeof(image_name), 12, SAMPLE_IMAGE1_PATH, i % 2) == -1) { /* 12 2 length */
            return TD_FAILURE;
        }
        dst = (td_u8*)viraddr;
        if (sample_gfbg_load_bmp(image_name, dst) != TD_SUCCESS) {
            return TD_FAILURE;
        }
        if (sample_quick_copy_by_tde(byte_per_pixel, tde_clr_fmt, fix_screen_stride, hide_screen_phy) !=
            TD_SUCCESS) {
            return TD_FAILURE;
        }

        var.yoffset = (i % 2 == 1) ? var.yres : 0; /* 2 for 0 1 */
        if (ioctl(info->fd, FBIOPAN_DISPLAY, &var) < 0) {
            sample_print("FBIOPAN_DISPLAY failed!\n");
            return TD_FAILURE;
        }
        sleep(1);
    }
    return TD_SUCCESS;
}

static td_s32 sample_move_cursor(pthread_gfbg_sample_info *info, struct fb_var_screeninfo *var, td_u8 *show_screen)
{
    td_s32 i;
    ot_fb_point point = {0, 0};

    point.x_pos = (info->ctrlkey == 3) ? 150 : 0; /* 3 150:for case;alg data */
    point.y_pos = (info->ctrlkey == 3) ? 150 : 0; /* 3 150:for case;alg data */

    if (sample_gfbg_load_bmp(SAMPLE_CURSOR_PATH, show_screen) != TD_SUCCESS) {
        return TD_FAILURE;
    }
    if (ioctl(info->fd, FBIOPAN_DISPLAY, var) < 0) {
        sample_print("FBIOPAN_DISPLAY failed!\n");
        return TD_FAILURE;
    }
    sample_print("show cursor\n");
    sleep(2); /* 2 second */
    for (i = 0; i < 100; i++) { /* 100 for times */
        if (g_exit_flag == 'q') {
            printf("process exit...\n");
            break;
        }
        point.x_pos += 2; /* 2 pos */
        point.y_pos += 2; /* 2 pos */
        if (ioctl(info->fd, FBIOPUT_SCREEN_ORIGIN_GFBG, &point) < 0) {
            sample_print("set screen original show position failed!\n");
            return TD_FAILURE;
        }
        usleep(70 * 1000); /* 70 1000 for sleep */
    }
    for (i = 0; i < 100; i++) { /* 100 for times */
        if (g_exit_flag == 'q') {
            printf("process exit...\n");
            break;
        }
        point.x_pos += 2; /* 2 pos */
        point.y_pos -= 2; /* 2 pos */
        if (ioctl(info->fd, FBIOPUT_SCREEN_ORIGIN_GFBG, &point) < 0) {
            sample_print("set screen original show position failed!\n");
            return TD_FAILURE;
        }
        usleep(70 * 1000); /* 70 1000 for sleep */
    }
    sample_print("move the cursor\n");
    sleep(1);
    return TD_SUCCESS;
}

static td_s32 sample_show_bitmap(pthread_gfbg_sample_info *info, td_u8 *show_screen, td_u32 fix_screen_stride,
    td_u32 byte_per_pixel)
{
    td_s32 ret;
    struct fb_var_screeninfo var;
    struct fb_fix_screeninfo fix;
    td_void *viraddr = TD_NULL;

    if (ioctl(info->fd, FBIOGET_VSCREENINFO, &var) < 0) {
        sample_print("get variable screen info failed!\n");
        return TD_FAILURE;
    }

    if (ioctl(info->fd, FBIOGET_FSCREENINFO, &fix) < 0) {
        sample_print("get fix screen info failed!\n");
        return TD_FAILURE;
    }

    switch (info->ctrlkey) {
        /* 2 means none buffer and just for pan display. */
        case 2:
            if (ss_mpi_sys_mmz_alloc(&g_phyaddr, ((void**)&viraddr), TD_NULL, TD_NULL, SAMPLE_IMAGE_WIDTH *
                SAMPLE_IMAGE_HEIGHT * byte_per_pixel) == TD_FAILURE) {
                sample_print("allocate memory (max_w*max_h*%u bytes) failed\n", byte_per_pixel);
                return TD_FAILURE;
            }
            ret = ss_tde_open();
            if (ret < 0) {
                sample_print("tde_open failed :%d!\n", ret);
                ss_mpi_sys_mmz_free(g_phyaddr, viraddr);
                g_phyaddr = 0;
                return TD_FAILURE;
            }
            if (sample_draw_line_and_picture(info, viraddr, show_screen, fix_screen_stride) != TD_SUCCESS) {
                ss_tde_close();
                ss_mpi_sys_mmz_free(g_phyaddr, viraddr);
                g_phyaddr = 0;
                return TD_FAILURE;
            }
            ss_mpi_sys_mmz_free(g_phyaddr, viraddr);
            g_phyaddr = 0;
            ss_tde_close();
            break;
        case 3: /* 3 mouse case */
            /* move cursor */
            if (sample_move_cursor(info, &var, show_screen) != TD_SUCCESS) {
                return TD_FAILURE;
            }
            break;
        default:
            return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_show_process(pthread_gfbg_sample_info *info)
{
    td_u8 *show_screen = TD_NULL;
    struct fb_var_screeninfo var;
    struct fb_fix_screeninfo fix;
    td_u32 fix_screen_stride;
    td_bool show;

    if (ioctl(info->fd, FBIOGET_VSCREENINFO, &var) < 0) {
        sample_print("get variable screen info failed!\n");
        goto ERR1;
    }

    if (ioctl(info->fd, FBIOGET_FSCREENINFO, &fix) < 0) {
        sample_print("get fix screen info failed!\n");
        goto ERR1;
    }
    fix_screen_stride = fix.line_length;
    show_screen = mmap(TD_NULL, fix.smem_len, PROT_READ | PROT_WRITE, MAP_SHARED, info->fd, 0);
    if (show_screen == MAP_FAILED) {
        sample_print("mmap framebuffer failed!\n");
        goto ERR1;
    }

    if (memset_s(show_screen, fix.smem_len, 0x0, fix.smem_len) != EOK) {
        goto ERR2;
    }

    if (sample_time_to_play(info, show_screen, fix_screen_stride) != TD_SUCCESS) {
        goto ERR2;
    }
    /* 8 for 1 byte */
    if (sample_show_bitmap(info, show_screen, fix_screen_stride, var.bits_per_pixel / 8) != TD_SUCCESS) {
        goto ERR2;
    }
    munmap(show_screen, fix.smem_len);
    show = TD_FALSE;
    if (ioctl(info->fd, FBIOPUT_SHOW_GFBG, &show) < 0) {
        sample_print("FBIOPUT_SHOW_GFBG failed!\n");
        close(info->fd);
        return TD_FAILURE;
    }
    close(info->fd);
    info->fd = -1;
    return TD_SUCCESS;

ERR2:
    munmap(show_screen, fix.smem_len);
    show_screen = TD_NULL;
ERR1:
    close(info->fd);
    info->fd = -1;
    return TD_FAILURE;
}

static td_void* sample_gfbg_pandisplay(td_void *data)
{
    pthread_gfbg_sample_info *info = TD_NULL;
    td_char thdname[64]; /* 64 for char length */
    td_char file[FILE_LENGTH_MAX] = {0};

    if (data == TD_NULL) {
        return TD_NULL;
    }
    info = (pthread_gfbg_sample_info*)data;
    if (snprintf_s(thdname, sizeof(thdname), 17, "GFBG%d_pandisplay", info->layer) == -1) { /* 17 for char length */
        printf("%s:%d:snprintf_s failed.\n", __FUNCTION__, __LINE__);
        return TD_NULL;
    }
    prctl(PR_SET_NAME, thdname, 0, 0, 0);

    if (sample_get_file_name(info, file, FILE_LENGTH_MAX) != TD_SUCCESS) {
        return TD_NULL;
    }

    if (sample_init_frame_buffer(info, file) != TD_SUCCESS) {
        return TD_NULL;
    }

    if (sample_init_var(info) != TD_SUCCESS) {
        close(info->fd);
        info->fd = -1;
        return TD_NULL;
    }

    /* map the physical video memory for user use */
    if (sample_show_process(info) != TD_SUCCESS) {
        return TD_NULL;
    }
    sample_print("[end]\n");
    return TD_NULL;
}

static td_s32 sample_init_frame_buffer_ex(pthread_gfbg_sample_info *info, const char *input_file)
{
    ot_fb_colorkey color_key;
    ot_fb_point point = {0, 0};
    td_char file[PATH_MAX + 1] = {0};

    if (strlen(input_file) > PATH_MAX || realpath(input_file, file) == TD_NULL) {
        return TD_FAILURE;
    }
    /* step 1. open framebuffer device overlay 0 */
    info->fd = open(file, O_RDWR, 0);
    if (info->fd < 0) {
        sample_print("open %s failed!\n", file);
        return TD_FAILURE;
    }

    /* all layer support colorkey */
    color_key.enable = TD_TRUE;
    color_key.value = 0x0;
    if (ioctl(info->fd, FBIOPUT_COLORKEY_GFBG, &color_key) < 0) {
        sample_print("FBIOPUT_COLORKEY_GFBG!\n");
        close(info->fd);
        info->fd = -1;
        return TD_FAILURE;
    }

    if (ioctl(info->fd, FBIOPUT_SCREEN_ORIGIN_GFBG, &point) < 0) {
        sample_print("set screen original show position failed!\n");
        close(info->fd);
        info->fd = -1;
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_init_var_ex(pthread_gfbg_sample_info *info)
{
    struct fb_var_screeninfo var;

    /* step 3. get the variable screen information */
    if (ioctl(info->fd, FBIOGET_VSCREENINFO, &var) < 0) {
        sample_print("get variable screen info failed!\n");
        return TD_FAILURE;
    }

    switch (info->color_format) {
        case OT_FB_FORMAT_ARGB8888:
            var.transp = g_a32;
            var.red = g_r32;
            var.green = g_g32;
            var.blue = g_b32;
            var.bits_per_pixel = 32; /* 32 for 4 byte */
            g_osd_color_fmt = OSD_COLOR_FORMAT_RGB8888;
            break;
        default:
            var.transp = g_a16;
            var.red = g_r16;
            var.green = g_g16;
            var.blue = g_b16;
            var.bits_per_pixel = 16; /* 16 for 2 byte */
            break;
    }

    var.xres_virtual = WIDTH_1080P;
    var.yres_virtual = HEIGHT_1080P;
    var.xres = WIDTH_1080P;
    var.yres = HEIGHT_1080P;
    var.activate = FB_ACTIVATE_NOW;

    /* step 5. set the variable screen information */
    if (ioctl(info->fd, FBIOPUT_VSCREENINFO, &var) < 0) {
        sample_print("put variable screen info failed!\n");
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_init_layer_info(pthread_gfbg_sample_info *info)
{
    ot_fb_layer_info layer_info = {0};
    switch (info->ctrlkey) {
        case 0: /* 0 case */
            layer_info.buf_mode = OT_FB_LAYER_BUF_ONE;
            layer_info.mask = OT_FB_LAYER_MASK_BUF_MODE;
            break;
        case 1: /* 1 case */
            layer_info.buf_mode = OT_FB_LAYER_BUF_DOUBLE;
            layer_info.mask = OT_FB_LAYER_MASK_BUF_MODE;
            break;
        default:
            layer_info.buf_mode = OT_FB_LAYER_BUF_NONE;
            layer_info.mask = OT_FB_LAYER_MASK_BUF_MODE;
    }
    if (ioctl(info->fd, FBIOPUT_LAYER_INFO, &layer_info) < 0) {
        sample_print("PUT_LAYER_INFO failed!\n");
        close(info->fd);
        info->fd = -1;
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_init_compress(pthread_gfbg_sample_info *info)
{
    td_bool show = TD_TRUE;
    if (ioctl(info->fd, FBIOPUT_SHOW_GFBG, &show) < 0) {
        sample_print("FBIOPUT_SHOW_GFBG failed!\n");
        close(info->fd);
        info->fd = -1;
        return TD_FAILURE;
    }

    if (info->compress == TD_TRUE) {
        if (ioctl(info->fd, FBIOPUT_COMPRESSION_GFBG, &info->compress) < 0) {
            sample_print("FBIOPUT_COMPRESSION_GFBG failed!\n");
            close(info->fd);
            info->fd = -1;
            return TD_FAILURE;
        }
    }
    return TD_SUCCESS;
}

static td_s32 sample_init_canvas(pthread_gfbg_sample_info *info, ot_fb_buf *canvas_buf, td_void **buf,
    td_void **viraddr)
{
    td_u32 byte_per_pixel;
    ot_fb_color_format clr_fmt;

    switch (info->color_format) {
        case OT_FB_FORMAT_ARGB8888:
            byte_per_pixel = 4; /* 4 bytes */
            clr_fmt = OT_FB_FORMAT_ARGB8888;
            break;
        default:
            byte_per_pixel = 2; /* 2 bytes */
            clr_fmt = OT_FB_FORMAT_ARGB1555;
            break;
    }

    if (ss_mpi_sys_mmz_alloc(&g_canvas_addr, buf, TD_NULL, TD_NULL, WIDTH_1080P * HEIGHT_1080P *
        (byte_per_pixel)) == TD_FAILURE) {
        sample_print("allocate memory (max_w*max_h*%u bytes) failed\n", byte_per_pixel);
        close(info->fd);
        info->fd = -1;
        return TD_FAILURE;
    }
    canvas_buf->canvas.phys_addr = g_canvas_addr;
    canvas_buf->canvas.height = HEIGHT_1080P;
    canvas_buf->canvas.width = WIDTH_1080P;
    canvas_buf->canvas.pitch = WIDTH_1080P * (byte_per_pixel);
    canvas_buf->canvas.format = clr_fmt;
    if (memset_s(*buf, WIDTH_1080P * HEIGHT_1080P * (byte_per_pixel), 0x00, canvas_buf->canvas.pitch *
        canvas_buf->canvas.height) != EOK) {
        sample_print("%s:%d:memset_s failed\n", __FUNCTION__, __LINE__);
        ss_mpi_sys_mmz_free(g_canvas_addr, *buf);
        g_canvas_addr = 0;
        close(info->fd);
        return TD_FAILURE;
    }

    /* change bmp */
    if (ss_mpi_sys_mmz_alloc(&g_phyaddr, viraddr, TD_NULL, TD_NULL, SAMPLE_IMAGE_WIDTH * SAMPLE_IMAGE_HEIGHT *
        byte_per_pixel) == TD_FAILURE) {
        sample_print("allocate memory (max_w*max_h*%u bytes) failed\n", byte_per_pixel);
        ss_mpi_sys_mmz_free(g_canvas_addr, *buf);
        g_canvas_addr = 0;
        close(info->fd);
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_draw_line_by_cpu_ex(pthread_gfbg_sample_info *info, ot_fb_buf *canvas_buf, td_void *buf)
{
    td_u32 x, y, color;
    td_s32 ret;
    ot_fb_color_format clr_fmt;

    switch (info->color_format) {
        case OT_FB_FORMAT_ARGB8888:
            clr_fmt = OT_FB_FORMAT_ARGB8888;
            color = GFBG_RED_8888;
            break;
        default:
            clr_fmt = OT_FB_FORMAT_ARGB1555;
            color = GFBG_RED_1555;
            break;
    }
    for (y = (HEIGHT_1080P / 2 - 2); y < (HEIGHT_1080P / 2 + 2); y++) { /* 2 alg data */
        for (x = 0; x < WIDTH_1080P; x++) {
            if (clr_fmt == OT_FB_FORMAT_ARGB8888) {
                *((td_u32*)buf + y * WIDTH_1080P + x) = color;
            } else {
                *((td_u16*)buf + y * WIDTH_1080P + x) = color;
            }
        }
    }
    for (y = 0; y < HEIGHT_1080P; y++) {
        for (x = (WIDTH_1080P / 2 - 2); x < (WIDTH_1080P / 2 + 2); x++) { /* 2 alg data */
            if (clr_fmt == OT_FB_FORMAT_ARGB8888) {
                *((td_u32*)buf + y * WIDTH_1080P + x) = color;
            } else {
                *((td_u16*)buf + y * WIDTH_1080P + x) = color;
            }
        }
    }
    canvas_buf->update_rect.x = 0;
    canvas_buf->update_rect.y = 0;
    canvas_buf->update_rect.width = WIDTH_1080P;
    canvas_buf->update_rect.height = HEIGHT_1080P;
    ret = ioctl(info->fd, FBIO_REFRESH, canvas_buf);
    if (ret < 0) {
        sample_print("REFRESH failed!\n");
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_void sample_source_cfg_tde(pthread_gfbg_sample_info *info, ot_tde_surface *src_surface, ot_tde_rect *src_rect,
    ot_tde_surface *dst_surface, ot_tde_rect *dst_rect)
{
    td_u32 byte_per_pixel;
    ot_tde_color_format tde_clr_fmt;

    switch (info->color_format) {
        case OT_FB_FORMAT_ARGB8888:
            tde_clr_fmt = OT_TDE_COLOR_FORMAT_ARGB8888;
            byte_per_pixel = 4; /* 4 bytes */
            break;
        default:
            tde_clr_fmt = OT_TDE_COLOR_FORMAT_ARGB1555;
            byte_per_pixel = 2; /* 2 bytes */
            break;
    }
    src_rect->pos_x   = 0;
    src_rect->pos_y   = 0;
    src_rect->height = SAMPLE_IMAGE_HEIGHT;
    src_rect->width  = SAMPLE_IMAGE_WIDTH;
    dst_rect->pos_x   = 0;
    dst_rect->pos_y   = 0;
    dst_rect->height = src_rect->height;
    dst_rect->width  = src_rect->width;

    dst_surface->color_format = tde_clr_fmt;
    dst_surface->width = WIDTH_1080P;
    dst_surface->height = HEIGHT_1080P;
    dst_surface->stride  = WIDTH_1080P * byte_per_pixel;

    src_surface->color_format = tde_clr_fmt;
    src_surface->width = SAMPLE_IMAGE_WIDTH;
    src_surface->height = SAMPLE_IMAGE_HEIGHT;
    src_surface->stride = byte_per_pixel * SAMPLE_IMAGE_WIDTH;
    src_surface->phys_addr = g_phyaddr;
    src_surface->support_alpha_ex_1555 = TD_TRUE;
    src_surface->alpha_max_is_255 = TD_TRUE;
    src_surface->alpha0 = 0XFF;
    src_surface->alpha1 = 0XFF;
    return;
}

static td_s32 sample_draw_picture_by_tde(pthread_gfbg_sample_info *info, ot_fb_buf *canvas_buf, td_u32 i,
    td_void *viraddr)
{
    td_char image_name[128]; /* 128 for char length */
    td_u8 *dst = TD_NULL;
    ot_tde_rect src_rect = {0};
    ot_tde_rect dst_rect = {0};
    ot_tde_surface src_surface = {0};
    ot_tde_surface dst_surface = {0};
    td_s32 handle;
    ot_tde_single_src single_src = {0};
    td_s32 ret;

    if (snprintf_s(image_name, sizeof(image_name), 12, SAMPLE_IMAGE1_PATH, i % 2) == -1) { /* 12 2 char length */
        sample_print("%s:%d:snprintf_s failed.\n", __FUNCTION__, __LINE__);
        return TD_FAILURE;
    }
    dst = (td_u8*)viraddr;
    if (sample_gfbg_load_bmp(image_name, dst) != TD_SUCCESS) {
        sample_print("sample_gfbg_load_bmp failed!\n");
        return TD_FAILURE;
    }

    sample_source_cfg_tde(info, &src_surface, &src_rect, &dst_surface, &dst_rect);
    dst_surface.phys_addr = canvas_buf->canvas.phys_addr;

    /* 1. start job */
    handle = ss_tde_begin_job();
    if (handle == OT_ERR_TDE_INVALID_HANDLE) {
        sample_print("start job failed!\n");
        return TD_FAILURE;
    }
    single_src.src_surface = &src_surface;
    single_src.dst_surface = &dst_surface;
    single_src.src_rect = &src_rect;
    single_src.dst_rect = &dst_rect;
    ret = ss_tde_quick_copy(handle, &single_src);
    if (ret < 0) {
        sample_print("tde_quick_copy:%d failed,ret=0x%x!\n", __LINE__, ret);
        ss_tde_cancel_job(handle);
        return TD_FAILURE;
    }

    /* 3. submit job */
    ret = ss_tde_end_job(handle, TD_FALSE, TD_TRUE, 10); /* timeout 10ms */
    if (ret < 0) {
        sample_print("line:%d,tde_end_job failed,ret=0x%x!\n", __LINE__, ret);
        ss_tde_cancel_job(handle);
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_time_to_play_ex(pthread_gfbg_sample_info *info, ot_fb_buf *canvas_buf, td_void *buf,
    td_void *viraddr)
{
    td_s32 ret;
    td_u32 i;

    ret = ss_tde_open();
    if (ret < 0) {
        sample_print("tde_open failed :%d!\n", ret);
        return TD_FAILURE;
    }

    sample_print("[begin]\n");
    sample_print("expected:two red line!\n");
    /* time to play */
    for (i = 0; i < SAMPLE_IMAGE_NUM; i++) {
        if (g_exit_flag == 'q') {
            printf("process exit...\n");
            break;
        }
        /* draw two lines by cpu */
        if (sample_draw_line_by_cpu_ex(info, canvas_buf, buf) != TD_SUCCESS) {
            return TD_FAILURE;
        }
        sleep(2); /* 2 second */

        if (sample_draw_picture_by_tde(info, canvas_buf, i, viraddr) != TD_SUCCESS) {
            return TD_FAILURE;
        }

        canvas_buf->update_rect.x = 0;
        canvas_buf->update_rect.y = 0;
        canvas_buf->update_rect.width = WIDTH_1080P;
        canvas_buf->update_rect.height = HEIGHT_1080P;
        ret = ioctl(info->fd, FBIO_REFRESH, canvas_buf);
        if (ret < 0) {
            sample_print("REFRESH failed!\n");
            return TD_FAILURE;
        }
        sleep(2); /* 2 second */
    }
    return TD_SUCCESS;
}

static td_void* sample_gfbg_refresh(td_void *data)
{
    td_char file[FILE_LENGTH_MAX] = {0};
    ot_fb_buf canvas_buf;
    td_void *buf = TD_NULL;
    td_void *viraddr = TD_NULL;
    pthread_gfbg_sample_info *info = TD_NULL;

    prctl(PR_SET_NAME, "GFBG_REFRESH", 0, 0, 0);
    if (data == TD_NULL) {
        return TD_NULL;
    }
    info = (pthread_gfbg_sample_info*)data;

    if (sample_get_file_name(info, file, FILE_LENGTH_MAX) != TD_SUCCESS) {
        return TD_NULL;
    }

    if (sample_init_frame_buffer_ex(info, file) != TD_SUCCESS) {
        return TD_NULL;
    }

    if (sample_init_var_ex(info) != TD_SUCCESS) {
        close(info->fd);
        info->fd = -1;
        return TD_NULL;
    }

    if (sample_init_layer_info(info) != TD_SUCCESS) {
        return TD_NULL;
    }

    if (sample_init_compress(info) != TD_SUCCESS) {
        return TD_NULL;
    }

    if (sample_init_canvas(info, &canvas_buf, &buf, &viraddr) != TD_SUCCESS) {
        return TD_NULL;
    }

    if (sample_time_to_play_ex(info, &canvas_buf, buf, viraddr) != TD_SUCCESS) {
        goto ERR;
    }
    sample_print("[end]\n");
ERR:
    ss_mpi_sys_mmz_free(g_phyaddr, viraddr);
    g_phyaddr = 0;
    ss_mpi_sys_mmz_free(g_canvas_addr, buf);
    g_canvas_addr = 0;
    close(info->fd);
    return TD_NULL;
}

static td_s32 sample_var_init(pthread_gfbg_sample_info *info, struct fb_var_screeninfo *var)
{
    if (info == TD_NULL) {
        return TD_FAILURE;
    }
    /* step 1. open framebuffer */
    info->fd = open("/dev/fb0", O_RDWR, 0);
    if (info->fd < 0) {
        sample_print("open /dev/fb0 failed!\n");
        return TD_FAILURE;
    }
    /* step 2. get the variable screen information */
    if (ioctl(info->fd, FBIOGET_VSCREENINFO, var) < 0) {
        sample_print("get variable screen info failed!\n");
        close(info->fd);
        return TD_FAILURE;
    }
    /* step 3. modify the variable screen info */
    switch (info->color_format) {
        case OT_FB_FORMAT_4BPP:
            var->transp = g_a4;
            var->red    = g_r4;
            var->green  = g_g4;
            var->blue   = g_b4;
            var->bits_per_pixel = 4; /* 4 bits per pixel */
            var->xres_virtual = WIDTH_1920;
            var->yres_virtual = HEIGHT_1080 * 2; /* alloc 2 buf */
            var->xres = WIDTH_1920;
            var->yres = HEIGHT_1080;
            var->activate = 0;
            var->xoffset = 0;
            var->yoffset = 0;
            break;
        default:
            return TD_FAILURE;
    }

    if (ioctl(info->fd, FBIOPUT_VSCREENINFO, var) < 0) {
        sample_print("put variable screen info failed!\n");
        close(info->fd);
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_cmap_init(pthread_gfbg_sample_info *info)
{
    struct fb_cmap cmap;
    cmap.start = 0;
    cmap.len = CMAP_LENGTH_MAX;
    cmap.red = g_cmap_red;
    cmap.green = g_cmap_green;
    cmap.blue = g_cmap_blue;
    cmap.transp = g_cmap_alpha;

    if (ioctl(info->fd, FBIOPUTCMAP, &cmap) < 0) {
        sample_print("put cmap info failed!\n");
        close(info->fd);
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_get_fix_and_mmap(pthread_gfbg_sample_info *info, struct fb_fix_screeninfo *fix,
                                      td_void **viraddr)
{
    if (info == TD_NULL || fix == TD_NULL || viraddr == TD_NULL) {
        return TD_FAILURE;
    }
    if (ioctl(info->fd, FBIOGET_FSCREENINFO, fix) < 0) {
        sample_print("get fix screen info failed!\n");
        close(info->fd);
        return TD_FAILURE;
    }

    *viraddr = mmap(TD_NULL, WIDTH_1920 * HEIGHT_1080,
                    PROT_READ | PROT_WRITE, MAP_SHARED, info->fd, 0);
    if (*viraddr == TD_NULL) {
        sample_print("mmap /dev/fb2 failed!\n");
        return TD_FAILURE;
    }

    return TD_SUCCESS;
}

static td_s32 sample_start_draw_rect(ot_tde_none_src *none_src, ot_tde_corner_rect_info *conner_rect)
{
    td_s32 handle;
    td_u32 i;
    ot_tde_corner_rect corner;

    handle = ss_tde_begin_job();
    if (handle == OT_ERR_TDE_INVALID_HANDLE) {
        return TD_FAILURE;
    }

    corner.corner_rect_region = none_src->dst_rect;
    corner.corner_rect_info = conner_rect;

    /* times draw rect,once commit */
    for (i = 0; i < 5; i++) { /* 5 draw 5 rect */
        if (ss_tde_draw_corner_box(handle, none_src->dst_surface, &corner, 1) < 0) {
            ss_tde_cancel_job(handle);
            return TD_FAILURE;
        }
        none_src->dst_rect->pos_x += 200; /* 200 alg data */
        none_src->dst_rect->pos_y += 200; /* 200 alg data */
    }

    if (ss_tde_end_job(handle, TD_FALSE, TD_TRUE, 10) < 0) { /* 10 timeout */
        ss_tde_cancel_job(handle);
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_draw_rect(td_void *viraddr, td_u32 index,
                               const struct fb_var_screeninfo *var, struct fb_fix_screeninfo fix)
{
    ot_tde_surface dst_surface = {0};
    ot_tde_rect dst_rect = {0};
    ot_tde_none_src none_src = {0};
    ot_tde_corner_rect_info conner_rect = {0};

    if (viraddr == TD_NULL) {
        return TD_FAILURE;
    }
    /* clear buffer,fill white */
    if (memset_s(viraddr, WIDTH_1920 * HEIGHT_1080, 0x33, WIDTH_1920 * HEIGHT_1080) != EOK) {
        sample_print("%s:%d:memset_s failed\n", __FUNCTION__, __LINE__);
        return TD_FAILURE;
    }
    if (ss_tde_open() < 0) {
        return TD_FAILURE;
    }

    dst_surface.color_format = OT_TDE_COLOR_FORMAT_CLUT4;
    dst_surface.width = WIDTH_1920;
    dst_surface.height = HEIGHT_1080;
    dst_surface.stride = fix.line_length;
    dst_surface.phys_addr = (index % 2) ? (fix.smem_start + fix.line_length * var->yres) : /* 2 alg data */
                            (fix.smem_start);
    dst_rect.pos_x = 0;
    dst_rect.pos_y = 0;
    dst_rect.height = 100; /* 100 alg data */
    dst_rect.width = 100; /* 100 alg data */
    none_src.dst_surface = &dst_surface;
    none_src.dst_rect = &dst_rect;
    /* conner rect */
    conner_rect.width = 10; /* 10 alg data */
    conner_rect.height = 20; /* 20 alg data */
    conner_rect.inner_color = 0x1;
    conner_rect.outer_color = (index % 2) ? 0x0 : 0x2; /* 2 alg data,0x2 for cmap index */

    if (sample_start_draw_rect(&none_src, &conner_rect) != TD_SUCCESS) {
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 sample_pandisplay(pthread_gfbg_sample_info *info, td_u32 index,
                                struct fb_var_screeninfo *var)
{
    if (info == TD_NULL || var == TD_NULL) {
        return TD_FAILURE;
    }
    if ((index % 2) == 0) { /* 2 alg data */
        var->yoffset = 0;
    } else {
        var->yoffset = var->yres;
    }

    if (ioctl(info->fd, FBIOPAN_DISPLAY, var) < 0) {
        sample_print("FBIOPAN_DISPLAY failed!\n");
        return TD_FAILURE;
    }

    sample_print("expected: buffer%d!\n", ((index % 2 == 0) ? 0 : 1)); /* 2 alg data */
    sample_print("wait 1 seconde.\n");
    sleep(1);
    ss_tde_close();
    return TD_SUCCESS;
}

static td_void *sample_gfbg_g0_clut(void *data)
{
    pthread_gfbg_sample_info *info = TD_NULL;
    struct fb_var_screeninfo var;
    struct fb_fix_screeninfo fix;
    td_s32 ret;
    td_void *viraddr = TD_NULL;
    td_u32 j = 0;

    if (data == TD_NULL) {
        return TD_NULL;
    }

    info = (pthread_gfbg_sample_info*)data;
    if (info->layer != GRAPHICS_LAYER_G0) {
        sample_print("++++%s:%d:only G0 support clut\n", __FUNCTION__, __LINE__);
        return TD_NULL;
    }

    ret = sample_var_init(info, &var);
    if (ret != TD_SUCCESS) {
        return TD_NULL;
    }

    ret = sample_cmap_init(info);
    if (ret != TD_SUCCESS) {
        return TD_NULL;
    }

    ret = sample_get_fix_and_mmap(info, &fix, &viraddr);
    if (ret != TD_SUCCESS) {
        return TD_NULL;
    }

    while (j < 20) { /* 20 times */
        if (sample_draw_rect(viraddr, j, &var, fix) != TD_SUCCESS) {
            munmap(viraddr, WIDTH_1920 * HEIGHT_1080);
            close(info->fd);
            return TD_NULL;
        }
        if (sample_pandisplay(info, j, &var) != TD_SUCCESS) {
            munmap(viraddr, WIDTH_1920 * HEIGHT_1080);
            close(info->fd);
            return TD_NULL;
        }
        j++;
    }
    munmap(viraddr, WIDTH_1920 * HEIGHT_1080);
    close(info->fd);
    sample_print("[end]\n");
    return TD_NULL;
}


static td_void sample_gfbg_handle_sig(td_s32 signo)
{
    static int sig_handled = 0;
    if (!sig_handled && (signo == SIGINT || signo == SIGTERM)) {
        sig_handled = 1;
        g_sample_gfbg_exit = 1;
    }
}

static td_void sample_gfbg_usage2(td_void)
{
    printf("\n\n/****************index******************/\n");
    printf("please choose the case which you want to run:\n");
    printf("\t0:  ARGB8888 standard mode\n");
    printf("\t1:  ARGB1555 BUF_DOUBLE mode\n");
    printf("\t2:  ARGB1555 BUF_ONE mode\n");
    printf("\t3:  ARGB1555 BUF_NONE mode\n");
    printf("\t4:  CLUT4 BUF_NONE mode\n");
    printf("\t5:  rotate mode\n");
    return;
}

static td_void sample_gfbg_usage1(td_char *s_prg_nm)
{
    printf("usage : %s <index> \n", s_prg_nm);
    sample_gfbg_usage2();
    return;
}

static td_s32 sample_gfbg_start_vo(vo_device_info *vo_dev_info)
{
    sample_print("shikeDebug start vo ...\n");
#if GFBG_BE_WITH_VO
    ot_vo_intf_type vo_intf_type = vo_dev_info->ot_vo_intf_type;
    g_vo_intf_type = vo_dev_info->ot_vo_intf_type;
    ot_vo_dev vo_dev = vo_dev_info->vo_dev;
    ot_vo_pub_attr pub_attr;
    td_u32  vo_frm_rate;
    ot_size size;
    td_s32 ret;
    sample_vo_cfg vo_config = {0};

    // sample_vo_cfg vo_config = {
    //     .vo_dev            = SAMPLE_VO_DEV_UHD,
    //     .vo_layer          = SAMPLE_VO_LAYER_VHD0,
    //     .vo_intf_type      = OT_VO_INTF_MIPI,
    //     .intf_sync         = OT_VO_OUT_1080P30,
    //     .bg_color          = COLOR_RGB_BLACK,
    //     .pix_format        = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_422,
    //     .disp_rect         = {0, 0, 1920, 1080},
    //     .image_size        = {1920, 1080},
    //     .vo_part_mode      = OT_VO_PARTITION_MODE_SINGLE,
    //     .dis_buf_len       = 3, /* 3: def buf len for single */
    //     .dst_dynamic_range = OT_DYNAMIC_RANGE_SDR8,
    //     .vo_mode           = VO_MODE_1MUX,
    //     .compress_mode     = OT_COMPRESS_MODE_NONE,
    // };

    /* step 1(start vo):  start vo device. */
    pub_attr.intf_type = vo_intf_type;
    pub_attr.intf_sync = OT_VO_OUT_1080P30;
    pub_attr.bg_color = COLOR_RGB_RED;
    ret = sample_comm_vo_get_width_height(pub_attr.intf_sync, &size.width, &size.height, &vo_frm_rate);
    if (ret != TD_SUCCESS) {
        sample_print("get vo width and height failed with %d!\n", ret);
        return ret;
    }
    ret = sample_comm_vo_start_dev(vo_dev, &pub_attr, &vo_config.user_sync, vo_config.dev_frame_rate);
    if (ret != TD_SUCCESS) {
        sample_print("start vo device failed with %d!\n", ret);
        return ret;
    }

    /* step 3(start vo): start bt1120 device. */
    /* if it's displayed on bt1120, we should start bt1120 */
    if (vo_intf_type & OT_VO_INTF_BT1120) {
        sample_comm_vo_bt1120_start(&pub_attr);
    }
    sample_print("shikeDebug start vo done\n");
    return TD_SUCCESS;
#else
    return TD_SUCCESS;
#endif
}

static td_void sample_gfbg_stop_vo(vo_device_info *vo_dev_info)
{
#if GFBG_BE_WITH_VO
    ot_vo_dev vo_dev = vo_dev_info->vo_dev;

    sample_comm_vo_stop_dev(vo_dev);
    return;
#else
    return;
#endif
}

static td_u32 sample_gfbg_vo_dev_get_layer(ot_vo_dev vo_dev)
{
    if (vo_dev == SAMPLE_VO_DEV_DHD0) {
        return GRAPHICS_LAYER_G0;
    } else {
        return GRAPHICS_LAYER_G1;
    }
}
static td_s32 sample_gfbg_standard_mode(vo_device_info *vo_dev_info)
{
    td_s32 ret;
    pthread_gfbg_sample_info info0;
    ot_vb_cfg vb_conf;

    /* step  1: init variable */
    if (memset_s(&vb_conf, sizeof(ot_vb_cfg), 0, sizeof(ot_vb_cfg)) != EOK) {
        sample_print("%s:%d:memset_s failed\n", __FUNCTION__, __LINE__);
        return TD_FAILURE;
    }
    /* step 2: mpp system init. */
    // shikeDebug
    // ret = sample_comm_sys_vb_init(&vb_conf);
    // if (ret != TD_SUCCESS) {
    //     sample_print("system init failed with %d!\n", ret);
    //     return ret;
    // }
    /*
     * step 3: start VO device.
     * NOTE: step 3 is optional when VO is running on other system.
     */
    ret = sample_gfbg_start_vo(vo_dev_info);
    if (ret != TD_SUCCESS) {
        sample_print("VO device %d start failed\n", vo_dev_info->vo_dev);
        goto sample_gfbg_standard_mode_0;
    }
    /* step 4:  start gfbg. */
    info0.layer = sample_gfbg_vo_dev_get_layer(vo_dev_info->vo_dev);
    info0.fd = -1;
    info0.ctrlkey = 2; /* 2 none buffer */
    info0.compress = TD_FALSE; /* compress opened or not */
    info0.color_format = OT_FB_FORMAT_ARGB8888;
    if (pthread_create(&g_gfbg_thread, 0, sample_gfbg_pandisplay, (td_void *)(&info0)) != 0) {
        sample_print("start gfbg thread0 failed!\n");
        goto sample_gfbg_standard_mode_1;
    }

    sample_gfbg_to_exit();
sample_gfbg_standard_mode_1:
    sample_gfbg_stop_vo(vo_dev_info);
sample_gfbg_standard_mode_0:
    
    //shikeDebug
    //sample_comm_sys_exit();
    return ret;
}

static td_s32 sample_gfbg_double_buf_mode(vo_device_info *vo_dev_info)
{
    td_s32 ret;
    pthread_gfbg_sample_info info0;
    g_vo_intf_type = vo_dev_info->ot_vo_intf_type;
    ot_vb_cfg vb_conf;

    /* step  1: init variable */
    if (memset_s(&vb_conf, sizeof(ot_vb_cfg), 0, sizeof(ot_vb_cfg)) != EOK) {
        sample_print("%s:%d:memset_s failed\n", __FUNCTION__, __LINE__);
        return TD_FAILURE;
    }

    /* step 2: mpp system init. */
    ret = sample_comm_sys_vb_init(&vb_conf);
    if (ret != TD_SUCCESS) {
        sample_print("system init failed with %d!\n", ret);
        return ret;
    }
    /*
     * step 3: start VO device.
     * NOTE: step 3 is optional when VO is running on other system.
     */
    ret = sample_gfbg_start_vo(vo_dev_info);
    if (ret != TD_SUCCESS) {
        sample_print("VO device %d start failed\n", vo_dev_info->vo_dev);
        goto sample_gfbg_double_buf_mode_0;
    }
    /* step 4:  start gfbg. */
    info0.layer = sample_gfbg_vo_dev_get_layer(vo_dev_info->vo_dev);
    info0.fd = -1;
    info0.ctrlkey = 1;   /* double buffer */
    info0.compress = TD_FALSE;
    info0.color_format = OT_FB_FORMAT_ABGR1555;
    if (pthread_create(&g_gfbg_thread, 0, sample_gfbg_refresh, (td_void*)(&info0)) != 0) {
        sample_print("start gfbg thread failed!\n");
        goto sample_gfbg_double_buf_mode_1;
    }
    sample_gfbg_to_exit();
sample_gfbg_double_buf_mode_1:
    sample_gfbg_stop_vo(vo_dev_info);
sample_gfbg_double_buf_mode_0:
    sample_comm_sys_exit();
    return ret;
}

static td_s32 sample_gfbg_one_buf_mode(vo_device_info *vo_dev_info)
{
    td_s32 ret;
    pthread_gfbg_sample_info info0;
    g_vo_intf_type = vo_dev_info->ot_vo_intf_type;
    ot_vb_cfg vb_conf;

    /* step  1: init variable */
    if (memset_s(&vb_conf, sizeof(ot_vb_cfg), 0, sizeof(ot_vb_cfg)) != EOK) {
        sample_print("%s:%d:memset_s failed\n", __FUNCTION__, __LINE__);
        return TD_FAILURE;
    }
    /* step 2: mpp system init. */
    ret = sample_comm_sys_vb_init(&vb_conf);
    if (ret != TD_SUCCESS) {
        sample_print("system init failed with %d!\n", ret);
        return ret;
    }
    /*
     * step 3: start VO device.
     * NOTE: step 3 is optional when VO is running on other system.
     */
    ret = sample_gfbg_start_vo(vo_dev_info);
    if (ret != TD_SUCCESS) {
        sample_print("VO device %d start failed\n", vo_dev_info->vo_dev);
        goto sample_gfbg_one_buf_mode_0;
    }
    /* step 4:  start gfbg. */
    info0.layer =  sample_gfbg_vo_dev_get_layer(vo_dev_info->vo_dev);
    info0.fd = -1;
    info0.ctrlkey = 0;   /* one buffer */
    info0.compress = TD_FALSE;
    info0.color_format = OT_FB_FORMAT_ABGR1555;
    if (pthread_create(&g_gfbg_thread, 0, sample_gfbg_refresh, (td_void*)(&info0)) != 0) {
        sample_print("start gfbg thread failed!\n");
        goto sample_gfbg_one_buf_mode_1;
    }
    sample_gfbg_to_exit();
sample_gfbg_one_buf_mode_1:
    sample_gfbg_stop_vo(vo_dev_info);
sample_gfbg_one_buf_mode_0:
    sample_comm_sys_exit();
    return ret;
}

static td_s32 sample_gfbg_none_buf_mode(vo_device_info *vo_dev_info)
{
    td_s32 ret;
    pthread_gfbg_sample_info info0;
    g_vo_intf_type = vo_dev_info->ot_vo_intf_type;
    ot_vb_cfg vb_conf;

    /* step  1: init variable */
    if (memset_s(&vb_conf, sizeof(ot_vb_cfg), 0, sizeof(ot_vb_cfg)) != EOK) {
        sample_print("%s:%d:memset_s failed\n", __FUNCTION__, __LINE__);
        return TD_FAILURE;
    }
    /* step 2: mpp system init. */
    ret = sample_comm_sys_vb_init(&vb_conf);
    if (ret != TD_SUCCESS) {
        sample_print("system init failed with %d!\n", ret);
        return ret;
    }
    /*
     * step 3: start VO device.
     * NOTE: step 3 is optional when VO is running on other system.
     */
    ret = sample_gfbg_start_vo(vo_dev_info);
    if (ret != TD_SUCCESS) {
        sample_print("VO device %d start failed\n", vo_dev_info->vo_dev);
        goto sample_gfbg_none_buf_mode_0;
    }
    /* step 4:  start gfbg. */
    info0.layer = sample_gfbg_vo_dev_get_layer(vo_dev_info->vo_dev);
    info0.fd = -1;
    info0.ctrlkey = 3; /* 3 0buffer refresh */
    info0.compress = TD_FALSE;
    info0.color_format = OT_FB_FORMAT_ABGR1555;
    if (pthread_create(&g_gfbg_thread, 0, sample_gfbg_refresh, (td_void*)(&info0)) != 0) {
        sample_print("start gfbg thread failed!\n");
        goto sample_gfbg_none_buf_mode_1;
    }
    sample_gfbg_to_exit();
sample_gfbg_none_buf_mode_1:
    sample_gfbg_stop_vo(vo_dev_info);
sample_gfbg_none_buf_mode_0:
    sample_comm_sys_exit();
    return ret;
}

static td_s32 sample_gfbg_clut_mode(vo_device_info *vo_dev_info)
{
    td_s32  ret;
    pthread_gfbg_sample_info info0;
    g_vo_intf_type  = vo_dev_info->ot_vo_intf_type;
    g_vo_dev = vo_dev_info->vo_dev;
    ot_vb_cfg  vb_conf;

    /* step  1: init variable */
    if (memset_s(&vb_conf, sizeof(ot_vb_cfg), 0, sizeof(ot_vb_cfg)) != EOK) {
        sample_print("%s:%d:memset_s failed\n", __FUNCTION__, __LINE__);
        return TD_FAILURE;
    }

    /* step 2: mpp system init. */
    ret = sample_comm_sys_vb_init(&vb_conf);
    if (ret != TD_SUCCESS) {
        sample_print("system init failed with %d!\n", ret);
        goto sample_gfbg_clut_mode_0;
    }

    /* step 3: start VO device.
     NOTE: step 3 is optional when VO is running on other system. */
    ret = sample_gfbg_start_vo(vo_dev_info);
    if (ret != TD_SUCCESS) {
        sample_print("VO device %d start failed\n", vo_dev_info->vo_dev);
        goto sample_gfbg_clut_mode_0;
    }

    /* step 4:  start gfbg. */
    info0.layer =  GRAPHICS_LAYER_G0;
    info0.fd  = -1;
    info0.ctrlkey = 2; /* 2 0buffer pan display */
    info0.compress = TD_FALSE;
    info0.color_format = OT_FB_FORMAT_4BPP;
    if (pthread_create(&g_gfbg_thread, 0, sample_gfbg_g0_clut, (void *)(&info0)) != 0) {
        sample_print("start gfbg thread failed!\n");
        goto sample_gfbg_clut_mode_1;
    }

    sample_gfbg_to_exit();

sample_gfbg_clut_mode_1:
    sample_gfbg_stop_vo(vo_dev_info);
sample_gfbg_clut_mode_0:
    sample_comm_sys_exit();

    return ret;
}

#ifdef CONFIG_SUPPORT_SAMPLE_ROTATION
static td_void gfbg_rotate(ot_vo_dev vo_dev)
{
    td_s32 ret;
    pthread_gfbg_sample_info info;

    info.layer = sample_gfbg_vo_dev_get_layer(vo_dev);
    info.ctrlkey = 1; /* 0 : one buf 1: double buf */
    info.compress = TD_FALSE;
    info.color_format = OT_FB_FORMAT_ARGB1555;

    info.fd = open("/dev/fb0", O_RDWR, 0);
    if (info.fd < 0) {
        sample_print("open /dev/fb0 failed!\n");
        return;
    }

    ret = gfbg_get_canvas(&info);
    if (ret != TD_SUCCESS) {
        close(info.fd);
        return;
    }

    ret = gfbg_draw(&info);
    if (ret != TD_SUCCESS) {
        gfbg_put_canvas();
        close(info.fd);
        return;
    }

    ret = gfbg_put_rotation(&info);
    if (ret != TD_SUCCESS) {
        gfbg_put_canvas();
        close(info.fd);
        return;
    }

    ret = gfbg_refresh(&info);
    if (ret != TD_SUCCESS) {
        gfbg_put_canvas();
        close(info.fd);
        return;
    }

    gfbg_put_canvas();
    close(info.fd);

    return;
}

static td_s32 gfbg_put_layer_info(pthread_gfbg_sample_info *info)
{
    td_s32 ret;
    ot_fb_layer_info layer_info = {0};

    ret = ioctl(info->fd, FBIOGET_LAYER_INFO, &layer_info);
    if (ret < 0) {
        sample_print("GET_LAYER_INFO failed!\n");
        return TD_FAILURE;
    }
    layer_info.mask = 0;
    layer_info.antiflicker_level = OT_FB_LAYER_ANTIFLICKER_AUTO;
    layer_info.mask |= OT_FB_LAYER_MASK_BUF_MODE;
    layer_info.mask |= OT_FB_LAYER_MASK_ANTIFLICKER_MODE;
    switch (info->ctrlkey) {
        case 0: /* 0 one buf */
            layer_info.buf_mode = OT_FB_LAYER_BUF_ONE;
            break;
        case 1: /* 1 double buf */
            layer_info.buf_mode = OT_FB_LAYER_BUF_DOUBLE;
            break;
        default:
            layer_info.buf_mode = OT_FB_LAYER_BUF_NONE;
    }

    ret = ioctl(info->fd, FBIOPUT_LAYER_INFO, &layer_info);
    if (ret < 0) {
        sample_print("PUT_LAYER_INFO failed!\n");
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 gfbg_get_canvas(pthread_gfbg_sample_info *info)
{
    td_s32 ret;
    td_u32 byte_per_pixel = (info->color_format == OT_FB_FORMAT_ARGB8888) ? 4 : 2; /* 4 2 bytes per pixel */
    ret = ss_mpi_sys_mmz_alloc(&g_canvas_phy, ((td_void**)&g_canvas_vir), TD_NULL, TD_NULL, WIDTH_800 *
        HEIGHT_600 * byte_per_pixel * 2); /* 2 double canvas buf */
    if (ret != TD_SUCCESS) {
        sample_print("allocate memory (maxW*maxH*%u bytes) failed\n", byte_per_pixel);
        return TD_FAILURE;
    }

    if ((g_canvas_phy == 0) || (g_canvas_vir == NULL)) {
        return TD_FAILURE;
    }
    (td_void)memset_s(g_canvas_vir, WIDTH_800 * HEIGHT_600 * byte_per_pixel * 2, 0xff, /* 2 double canvas buf */
        WIDTH_800 * HEIGHT_600 * byte_per_pixel * 2); /* 2 double canvas buf */
    ret = ss_mpi_sys_mmz_alloc(&g_picture_phy, ((td_void**)&g_picture_vir), NULL, NULL, SAMPLE_IMAGE_WIDTH *
        SAMPLE_IMAGE_HEIGHT * byte_per_pixel);
    if (ret != TD_SUCCESS) {
        sample_print("allocate memory (maxW*maxH*%d bytes) failed\n", byte_per_pixel);
        return TD_FAILURE;
    }
    if ((g_picture_phy == 0) || (g_picture_vir == NULL)) {
        return TD_FAILURE;
    }
    (td_void)memset_s(g_picture_vir, SAMPLE_IMAGE_WIDTH * SAMPLE_IMAGE_HEIGHT * byte_per_pixel, 0xff,
        SAMPLE_IMAGE_WIDTH * SAMPLE_IMAGE_HEIGHT * byte_per_pixel);
    return TD_SUCCESS;
}

static td_void gfbg_put_canvas(td_void)
{
    if ((g_canvas_phy == 0) || (g_canvas_vir == NULL) || (g_picture_phy == 0) || (g_picture_vir == NULL)) {
        return;
    }

    ss_mpi_sys_mmz_free(g_canvas_phy, g_canvas_vir);
    g_canvas_phy = 0;
    g_canvas_vir = NULL;
    ss_mpi_sys_mmz_free(g_picture_phy, g_picture_vir);
    g_picture_phy = 0;
    g_picture_vir = NULL;

    return;
}

static td_void gfbg_init_surface(ot_tde_surface *src_surface, ot_tde_surface *dst_surface,
    ot_tde_rect *src_rect, ot_tde_rect *dst_rect, td_u32 byte_per_pixel)
{
    src_rect->pos_x = dst_rect->pos_x = 0;
    src_rect->pos_y = dst_rect->pos_y = 0;
    src_rect->height = dst_rect->height = SAMPLE_IMAGE_HEIGHT;
    src_rect->width = dst_rect->width = SAMPLE_IMAGE_WIDTH;
    src_surface->color_format = dst_surface->color_format = OT_TDE_COLOR_FORMAT_ARGB1555;
    src_surface->width = SAMPLE_IMAGE_WIDTH;
    src_surface->height = SAMPLE_IMAGE_HEIGHT;
    src_surface->stride = byte_per_pixel * SAMPLE_IMAGE_WIDTH;
    src_surface->phys_addr = g_picture_phy;
    src_surface->support_alpha_ex_1555 = dst_surface->support_alpha_ex_1555 = TD_TRUE;
    src_surface->alpha_max_is_255 = dst_surface->alpha_max_is_255 = TD_TRUE;
    src_surface->alpha0 = dst_surface->alpha0 = 0XFF;
    src_surface->alpha1 = dst_surface->alpha1 = 0XFF;
    dst_surface->width = WIDTH_800;
    dst_surface->height = HEIGHT_600;
    dst_surface->stride = WIDTH_800 * byte_per_pixel;
    dst_surface->phys_addr = g_canvas_phy;
    dst_surface->is_ycbcr_clut = TD_FALSE;
    return;
}

static td_s32 gfbg_draw_by_tde(ot_tde_surface *src_surface, ot_tde_surface *dst_surface,
    ot_tde_rect *src_rect, ot_tde_rect *dst_rect, td_s32 index)
{
    td_s32 ret, handle;
    ot_tde_single_src single_src = {0};
    if (index == 1) {
        sample_gfbg_load_bmp("./res/1.bmp", g_picture_vir);
    } else {
        sample_gfbg_load_bmp("./res/0.bmp", g_picture_vir);
        dst_surface->phys_addr = g_canvas_phy + dst_surface->stride * dst_surface->height;
    }

    /* 1. start job */
    handle = ss_tde_begin_job();
    if (handle == OT_ERR_TDE_INVALID_HANDLE) {
        sample_print("start job failed!\n");
        return TD_FAILURE;
    }
    single_src.src_surface = src_surface;
    single_src.src_rect = src_rect;
    single_src.dst_surface = dst_surface;
    single_src.dst_rect = dst_rect;
    ret = ss_tde_quick_copy(handle, &single_src);
    if (ret < 0) {
        sample_print("tde_quick_copy:%d failed,ret=0x%x!\n", __LINE__, ret);
        ss_tde_cancel_job(handle);
        return TD_FAILURE;
    }

    /* 3. submit job */
    ret = ss_tde_end_job(handle, TD_FALSE, TD_TRUE, 10000); /* 10000 timeout */
    if (ret < 0) {
        sample_print("Line:%d,tde_end_job failed,ret=0x%x!\n", __LINE__, ret);
        ss_tde_cancel_job(handle);
        return TD_FAILURE;
    }
    return TD_SUCCESS;
}

static td_s32 gfbg_draw(pthread_gfbg_sample_info *info)
{
    td_s32 ret;
    ot_tde_rect dst_rect, src_rect;
    ot_tde_surface dst_surface = {0};
    ot_tde_surface src_surface = {0};
    td_u32 byte_per_pixel = (info->color_format == OT_FB_FORMAT_ARGB8888) ? 4 : 2; /* 4 2 bytes per pixel */

    ret = ss_tde_open();
    if (ret < 0) {
        sample_print("tde_open failed :%d!\n", ret);
        return TD_FAILURE;
    }

    gfbg_init_surface(&src_surface, &dst_surface, &src_rect, &dst_rect, byte_per_pixel);

    ret = gfbg_draw_by_tde(&src_surface, &dst_surface, &src_rect, &dst_rect, 1);
    if (ret != TD_SUCCESS) {
        return TD_FAILURE;
    }
    ret = gfbg_draw_by_tde(&src_surface, &dst_surface, &src_rect, &dst_rect, 0);
    if (ret != TD_SUCCESS) {
        return TD_FAILURE;
    }

    ss_tde_close();
    return TD_SUCCESS;
}

static td_s32 gfbg_refresh(const pthread_gfbg_sample_info *info)
{
    td_s32 ret;
    ot_fb_buf canvas_buf;
    td_u32 byte_per_pixel;
    td_s32 i;

    switch (info->color_format) {
        case OT_FB_FORMAT_ARGB8888:
            byte_per_pixel = 4; /* 4 bytes */
            break;
        default:
            byte_per_pixel = 2; /* 2 bytes */
            break;
    }
    canvas_buf.canvas.phys_addr = g_canvas_phy;
    canvas_buf.canvas.height = HEIGHT_600;
    canvas_buf.canvas.width = WIDTH_800;
    canvas_buf.canvas.pitch = WIDTH_800 * (byte_per_pixel);
    canvas_buf.canvas.format = info->color_format;
    canvas_buf.update_rect.x = 0;
    canvas_buf.update_rect.y = 0;
    canvas_buf.update_rect.width = WIDTH_800;
    canvas_buf.update_rect.height = HEIGHT_600;

    for (i = 0; i < 10; i++) { /* 10 times */
        if (i % 2 != 0) { /* 2 change 0.bmp or 1.bmp */
            canvas_buf.canvas.phys_addr = g_canvas_phy + canvas_buf.canvas.pitch * canvas_buf.canvas.height;
        } else {
            canvas_buf.canvas.phys_addr = g_canvas_phy;
        }
        ret = ioctl(info->fd, FBIO_REFRESH, &canvas_buf);
        if (ret < 0) {
            sample_print("REFRESH failed!\n");
            return TD_FAILURE;
        }

        sample_print("wait 1 seconds\n");
        usleep(1 * 1000 * 1000); /* 1000 1seconds */
    }

    return TD_SUCCESS;
}

static td_void gfbg_get_var_by_format(pthread_gfbg_sample_info *info, struct fb_var_screeninfo *var_info)
{
    switch (info->color_format) {
        case OT_FB_FORMAT_ARGB8888:
            var_info->transp = g_a32;
            var_info->red    = g_r32;
            var_info->green  = g_g32;
            var_info->blue   = g_b32;
            var_info->bits_per_pixel = 32; /* 32 bits */
            break;
        default:
            var_info->transp = g_a16;
            var_info->red    = g_r16;
            var_info->green  = g_g16;
            var_info->blue   = g_b16;
            var_info->bits_per_pixel = 16; /* 16 bits */
            break;
    }
}

static td_void gfbg_put_rotation_degree(struct fb_var_screeninfo *var_info, ot_fb_rotate_mode *rotate_mode)
{
    td_char input;
    printf("\n\n/****************index******************/\n");
    printf("please input 1 or 2 or 3 to choose the case which you want to run:\n");
    printf("\t1:  rotate 90\n");
    printf("\t2:  rotate 180\n");
    printf("\t3:  rotate 270\n");

    input = sample_gfbg_getchar();
    if (input == '1') {
        var_info->xres = var_info->xres_virtual = HEIGHT_600;
        var_info->yres = var_info->yres_virtual = WIDTH_800;
        *rotate_mode = OT_FB_ROTATE_90;
    } else if (input == '2') {
        var_info->xres = var_info->xres_virtual = WIDTH_800;
        var_info->yres = var_info->yres_virtual = HEIGHT_600;
        *rotate_mode = OT_FB_ROTATE_180;
    } else if (input == '3') {
        var_info->xres = var_info->xres_virtual = HEIGHT_600;
        var_info->yres = var_info->yres_virtual = WIDTH_800;
        *rotate_mode = OT_FB_ROTATE_270;
    } else {
        var_info->xres = var_info->xres_virtual = WIDTH_800;
        var_info->yres = var_info->yres_virtual = HEIGHT_600;
        *rotate_mode = OT_FB_ROTATE_NONE;
        sample_print("input param invalid, no rotate!\n");
    }
}

static td_s32 gfbg_put_rotation(pthread_gfbg_sample_info *info)
{
    td_s32 ret;
    struct fb_var_screeninfo var_info = {0};
    ot_fb_rotate_mode rotate_mode = OT_FB_ROTATE_BUTT;

    ret = ioctl(info->fd, FBIOGET_VSCREENINFO, &var_info);
    if (ret < 0) {
        sample_print("FBIOGET_VSCREENINFO failed!\n");
        return TD_FAILURE;
    }

    gfbg_get_var_by_format(info, &var_info);

    gfbg_put_rotation_degree(&var_info, &rotate_mode);

    ret = ioctl(info->fd, FBIOPUT_VSCREENINFO, &var_info);
    if (ret < 0) {
        sample_print("PUT_VSCREENINFO failed!\n");
        return TD_FAILURE;
    }

    ret = gfbg_put_layer_info(info);
    if (ret != TD_SUCCESS) {
        sample_print("PUT_LAYER_INFO failed!\n");
        return TD_FAILURE;
    }

    ret = ioctl(info->fd, FBIOPUT_ROTATE_MODE, &rotate_mode);
    if (ret < 0) {
        sample_print("rotate failed!\n");
        return TD_FAILURE;
    }

    return TD_SUCCESS;
}

static td_s32 sample_gfbg_rotation(vo_device_info *vo_dev_info)
{
    td_s32 ret;
    ot_vb_cfg vb_conf;

    if (memset_s(&vb_conf, sizeof(ot_vb_cfg), 0, sizeof(ot_vb_cfg)) != EOK) {
        sample_print("%s:%d:memset_s failed\n", __FUNCTION__, __LINE__);
        return TD_FAILURE;
    }
    ret = sample_comm_sys_vb_init(&vb_conf);
    if (ret != TD_SUCCESS) {
        sample_print("system init failed with %d!\n", ret);
        goto sample_gfbg_rotation_0;
    }

    /* open display */
    ret = sample_gfbg_start_vo(vo_dev_info);
    if (ret != TD_SUCCESS) {
        sample_print("VO device %d start failed\n", vo_dev_info->vo_dev);
        goto sample_gfbg_rotation_0;
    }

    gfbg_rotate(vo_dev_info->vo_dev);

    sample_gfbg_stop_vo(vo_dev_info);
sample_gfbg_rotation_0:
    sample_comm_sys_exit();
    return (ret);
}
#endif

static td_void sample_gfbg_to_exit_signal(td_void)
{
    printf("\033[0;31mreceive the signal,wait......!\033[0;39m\n");
    if (g_gfbg_thread1) {
        pthread_join(g_gfbg_thread1, 0);
        g_gfbg_thread1 = 0;
    }

    if (g_gfbg_thread) {
        pthread_join(g_gfbg_thread, 0);
        g_gfbg_thread = 0;
    }

    if (g_phyaddr) {
        ss_mpi_sys_mmz_free(g_phyaddr, TD_NULL);
        g_phyaddr = 0;
    }

    if (g_canvas_addr) {
        ss_mpi_sys_mmz_free(g_canvas_addr, TD_NULL);
        g_canvas_addr = 0;
    }

    if (g_canvas_phy) {
        ss_mpi_sys_mmz_free(g_canvas_phy, TD_NULL);
        g_canvas_phy = 0;
    }

    if (g_picture_phy) {
        ss_mpi_sys_mmz_free(g_picture_phy, TD_NULL);
        g_picture_phy = 0;
    }
    return;
}

static td_s32 sample_choose_the_case(char **argv, vo_device_info *vo_dev_info)
{
    td_s32 ret = TD_FAILURE;
    td_char ch;
    ch = *(argv[1]);
    g_exit_flag = 0;
    if (ch == '0') {
        sample_print("\nindex 0 selected.\n");
        ret = sample_gfbg_standard_mode(vo_dev_info);
    } else if (ch == '1') {
        sample_print("\nindex 1 selected.\n");
        ret = sample_gfbg_double_buf_mode(vo_dev_info);
    } else if (ch == '2') {
        sample_print("\nindex 2 selected.\n");
        ret = sample_gfbg_one_buf_mode(vo_dev_info);
    } else if (ch == '3') {
        sample_print("\nindex 3 selected.\n");
        ret = sample_gfbg_none_buf_mode(vo_dev_info);
    } else if (ch == '4') {
        sample_print("\nindex 4 selected.\n");
        ret = sample_gfbg_clut_mode(vo_dev_info);
#ifdef CONFIG_SUPPORT_SAMPLE_ROTATION
    } else if (ch == '5') {
        sample_print("\nindex 5 selected.\n");
        ret = sample_gfbg_rotation(vo_dev_info);
#endif
    } else {
        printf("index invalid! please try again.\n");
        sample_gfbg_usage1(argv[0]);
        return TD_FAILURE;
    }
    if (ret == TD_SUCCESS) {
        sample_print("program exit normally!\n");
    } else {
        sample_print("program exit abnormally!\n");
    }
    return ret;
}

#ifdef __LITEOS__
#define SAMPLE_GFBG_NAME "sample"
void sample_vo_sel_usage(td_void)
{
    printf("usage : %s <index> \n", SAMPLE_GFBG_NAME);
    sample_gfbg_usage2();
    return;
}
#endif

#ifdef __LITEOS__
int app_main(int argc, char *argv[])
{
    td_s32 ret;
    vo_device_info vo_dev_info;
    vo_dev_info.vo_dev = SAMPLE_VO_DEV_DHD0;
    vo_dev_info.ot_vo_intf_type = OT_VO_INTF_BT1120; /* default: BT1120 */

    if ((argc != GFBG_INPUT_PARAMETERS_NUM) || (strlen(argv[1]) != 1)) {
        sample_vo_sel_usage();
        return TD_FAILURE;
    }

    ret = sample_choose_the_case(argv, &vo_dev_info);
    if (ret != TD_SUCCESS) {
        return ret;
    }
    return ret;
}
#else
int main(int argc, char *argv[])
{
    td_s32 ret;
    vo_device_info vo_dev_info;
    vo_dev_info.vo_dev = SAMPLE_VO_DEV_UHD; //SAMPLE_VO_DEV_DHD0;
    vo_dev_info.ot_vo_intf_type = OT_VO_INTF_MIPI; /* default:BT1120 */
    //OT_VO_INTF_MIPI OT_VO_INTF_RGB_8BIT 

    if ((argc != GFBG_INPUT_PARAMETERS_NUM) || (strlen(argv[1]) != 1)) {
        printf("index invalid! please try again.\n");
        sample_gfbg_usage1(argv[0]);
        return TD_FAILURE;
    }

    sample_sys_signal(&sample_gfbg_handle_sig);

    ret = sample_choose_the_case(argv, &vo_dev_info);
    if (ret != TD_SUCCESS) {
        return ret;
    }
    return ret;
}
#endif
