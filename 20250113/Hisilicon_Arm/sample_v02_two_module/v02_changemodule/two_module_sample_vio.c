/*
  Copyright (c), 2001-2024, Shenshu Tech. Co., Ltd.
 */
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
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
#include <sys/prctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>

#include "sample_comm.h"
#include "sample_ipc.h"
#include "securec.h"
#include "ss_mpi_ae.h"
#include "uart_camera_ctrl.h"
#include "zoom_inout_thread.h"
#include "IIC_GPIO.h"
#include "geodetic_calculation.h"

#include "osd.h"

#define X_ALIGN 16
#define Y_ALIGN 2
#define out_ratio_1(x) ((x) / 3)
#define out_ratio_2(x) ((x) * 2 / 3)
#define out_ratio_3(x) ((x) / 2)

static volatile sig_atomic_t g_sig_flag = 0;

#define RGN_HANDLE_NUM_8 8
#define MM_BMP "./res/mm.bmp"
td_s32 sample_destroy_region(td_void);

//=============================================================================================================================================
// 电子罗盘
//=============================================================================================================================================
#include "serial.h"
#include "wit_c_sdk.h"
#include "REG.h"

#define ACC_UPDATE		0x01
#define GYRO_UPDATE		0x02
#define ANGLE_UPDATE	0x04
#define MAG_UPDATE		0x08
#define READ_UPDATE		0x80

static volatile char s_cDataUpdate = 0;
static int s_iCurBaud = 9600;
static void SensorDataUpdata(uint32_t uiReg, uint32_t uiRegNum);
static void AutoScanSensor(char* dev);
const int c_uiBaud[] = {2400 , 4800 , 9600 , 19200 , 38400 , 57600 , 115200 , 230400 , 460800 , 921600};

#define I2C_NAME_SIZE 0X80
// 电子罗盘串口文件符
int e_compass_fd = -1;
// 运行标志
volatile int e_compass_running = 1;           
// 电子罗盘串口UART操作互斥锁 
pthread_t e_compass_uart_thread;  

//================================================================================================================================================
// GPS
//================================================================================================================================================
#include "nmeaparser.h"

#define GPS_INFO_BUFFER_SIZE 256
#define I2C_NAME_SIZE 0X80
// GPS定位串口文件符
int e_GPS_fd = -1;
// 运行标志
volatile int e_GPS_running = 1;           
// GPS定位串口UART操作互斥锁 
pthread_t e_GPS_uart_thread;  
struct nmea_parser parser[1];

//=================================================================================================================================================
// distance measure
//=================================================================================================================================================

#define MEASURE_DISTANCE_BUFFER_SIZE 256
// 亮源激光串口文件符
int ly_measure_fd = -1;
// 运行标志
volatile int ly_measure_running = 1;           
// 亮源激光串口UART操作互斥锁
pthread_mutex_t ly_measure_uart_mutex;      
pthread_t ly_measure_uart_thread;  

//=================================================================================================================================================
// OSD 字符
//=================================================================================================================================================
// 静态全局变量存储传感器数据
static struct {
    double latitude;    // 纬度
    double longitude;   // 经度
    double height;      //海拔
    int GPS_status;
    double fAngle_x;// x-罗盘
    double fAngle_y;// y-罗盘
    double fAngle_z;// z-罗盘
    int Compass_status;
    double distance_m;// 激光测距
    double now_latitude;    
    double now_longitude;  
    double now_height;      
    double now_fAngle_x;
    double now_fAngle_y;
    double now_fAngle_z;
    int Distance_status;
} osd_data = {0};
pthread_t  osd_tid;
static sem_t g_data_ready;  // 信号量用于同步,数据就绪信号量
static pthread_mutex_t g_lock;  // 互斥锁
volatile bool update_osd_required ; // 检查是否有新OSD数据需要显示
// 创建OSD字符
static ot_bmp stBitmap;

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
    .intf_sync         = OT_VO_OUT_1280x1024_60,
    .bg_color          = COLOR_RGB_BLACK,
    .pix_format        = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420,
    .disp_rect         = {0, 0, 1280, 1024},
    .image_size        = {1280, 1024},
    .vo_part_mode      = OT_VO_PARTITION_MODE_SINGLE,
    .dis_buf_len       = 3, /* 3: def buf len for single */
    .dst_dynamic_range = OT_DYNAMIC_RANGE_SDR8,
    .vo_mode           = VO_MODE_1MUX,
    .compress_mode     = OT_COMPRESS_MODE_SEG,
};

sample_mipi_tx_config g_mipi_tx_config = {
    .intf_sync = OT_MIPI_TX_OUT_1280X1024_60,
};

static sample_comm_venc_chn_param g_venc_chn_param = {
    .frame_rate           = 30, /* 30 is a number */
    .stats_time           = 2,  /* 2 is a number */
    .gop                  = 60, /* 60 is a number */
    .venc_size            = {1280, 1024},
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

static td_s32 sample_vio_start_vpss(sample_sns_type sns_type,ot_vpss_grp grp, sample_vpss_cfg *vpss_cfg)
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
    ret = sample_common_vpss_start(sns_type,grp, &vpss_cfg->grp_attr, &vpss_chn_attr);
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

//=============================================================================================================================================
// 电子罗盘
//=============================================================================================================================================

static void SensorDataUpdata(uint32_t uiReg, uint32_t uiRegNum)
{
    int i;
    for(i = 0; i < uiRegNum; i++)
    {
        switch(uiReg)
        {
//            case AX:
//            case AY:
            case AZ:
				s_cDataUpdate |= ACC_UPDATE;
            break;
//            case GX:
//            case GY:
            case GZ:
				s_cDataUpdate |= GYRO_UPDATE;
            break;
//            case HX:
//            case HY:
            case HZ:
                s_cDataUpdate |= MAG_UPDATE;
            break;
//            case Roll:static int fd, s_iCurBaud = 9600;
//            case const int c_uiBaud[] = {2400 , 4800 , 9600 , 19200 , 38400 , 57600 , 115200 , 230400 , 460800 , 921600};Pitch:
            case Yaw:
				s_cDataUpdate |= ANGLE_UPDATE;
            break;
            default:
				s_cDataUpdate |= READ_UPDATE;
			break;
        }
		uiReg++;
    }
}

static void AutoScanSensor(char* dev)
{
	int i, iRetry;
	char cBuff[1];
	
	for(i = 1; i < 10; i++)
	{
		serial_close(e_compass_fd);
		s_iCurBaud = c_uiBaud[i];
		e_compass_fd = serial_open(dev , c_uiBaud[i]);
		
		iRetry = 2;
		do
		{
			s_cDataUpdate = 0;
			WitReadReg(AX, 3);
			usleep(200000);
			while(serial_read_data(e_compass_fd, cBuff, 1))
			{
				WitSerialDataIn(cBuff[0]);
			}
			if(s_cDataUpdate != 0)
			{
				printf("%d baud find sensor\r\n\r\n", c_uiBaud[i]);
				return ;
			}
			iRetry--;
		}while(iRetry);		
	}
	printf("can not find sensor\r\n");
	printf("please check your connection\r\n");
}

// 串口读取线程函数
void *e_compass_uart_read_method(void *arg) {
    float fAcc[3], fGyro[3], fAngle[3];
    int i;
	char cBuff[1];

    printf("e compass UART read thread started\n");
    
    while (e_compass_running) {
        
        while(serial_read_data(e_compass_fd, cBuff, 1))
        {
            WitSerialDataIn(cBuff[0]);
        }
        usleep(50000);

        if(s_cDataUpdate)
        {
            for(i = 0; i < 3; i++)
            {
                fAcc[i] = sReg[AX+i] / 32768.0f * 16.0f;
                fGyro[i] = sReg[GX+i] / 32768.0f * 2000.0f;
                fAngle[i] = sReg[Roll+i] / 32768.0f * 180.0f;
            }

            if(s_cDataUpdate & ACC_UPDATE)
            {
                //printf("acc:%.3f %.3f %.3f\r\n", fAcc[0], fAcc[1], fAcc[2]);
                s_cDataUpdate &= ~ACC_UPDATE;
            }
            if(s_cDataUpdate & GYRO_UPDATE)
            {
                //printf("gyro:%.3f %.3f %.3f\r\n", fGyro[0], fGyro[1], fGyro[2]);
                s_cDataUpdate &= ~GYRO_UPDATE;
            }
            if(s_cDataUpdate & ANGLE_UPDATE)
            {
                pthread_mutex_lock(&g_lock);
                osd_data.fAngle_x = fAngle[0];
                osd_data.fAngle_y = fAngle[1];
                osd_data.fAngle_z = fAngle[2];
                if (fAngle[0] == 0 && fAngle[1] == 0 && fAngle[2] == 0) {
                    osd_data.Compass_status = 0;
                } else {
                    osd_data.Compass_status = 1;
                }
                pthread_mutex_unlock(&g_lock);
                sem_post(&g_data_ready); // 释放信号量通知显示线程
                
                //printf("angle:%.3f %.3f %.3f\r\n", fAngle[0], fAngle[1], fAngle[2]);
                s_cDataUpdate &= ~ANGLE_UPDATE;
            }
            if(s_cDataUpdate & MAG_UPDATE)
            {
                //printf("mag:%d %d %d\r\n", sReg[HX], sReg[HY], sReg[HZ]);
                s_cDataUpdate &= ~MAG_UPDATE;
            }
        }
    }
    
    printf("e compass UART read thread exiting\n");
    return NULL;
}

int Open_Compass_Measure(){
    //打开电子罗盘测距串口
    char file_name[I2C_NAME_SIZE];
    if (sprintf_s(file_name, I2C_NAME_SIZE, "/dev/ttyAMA4") < 0) {
        printf("e compass sprintf_s error!\n");
        return -1;
    }

    if((e_compass_fd = serial_open(file_name , 9600)<0))
    {
        printf("open %s fail\n", file_name);
        return 0;
    }
    else printf("open %s success\n", file_name);

    printf("e compass set UART success.\n");

    // 电子罗盘寄存器
    WitInit(WIT_PROTOCOL_NORMAL, 0x50);
    WitRegisterCallBack(SensorDataUpdata);

    AutoScanSensor(file_name);

    //读数据的串口进程
    if(pthread_create(&e_compass_uart_thread, NULL, e_compass_uart_read_method, NULL) != 0) {
        printf("e compass failed to create UART read thread");
        close(e_compass_fd);
        return -1;
    }

    return 0;
}

int Close_Compass_Measure(){

    // 清理资源
    e_compass_running = 0;
    pthread_join(e_compass_uart_thread, NULL);
    close(e_compass_fd);
    printf("e compass program exited successfully\n");
    return 0;
}

//================================================================================================================================================
// GPS
//================================================================================================================================================

/*
 * Print navigation data and wait for user's keystroke
 * @navdata:    the navigation data
 */
void display_navdata(struct nav_data *navdata) {
    // printf("NAVDATA:\n");
    // printf("NAVDATA.FIX_VALID = %d\n", navdata->is_fixed);
    // printf("NAVDATA.DATE = %d-%02d-%02d\n", navdata->date.year, navdata->date.month, navdata->date.day);
    // printf("NAVDATA.TIME= %02d:%02d:%02d.%03d\n", navdata->time.hour, navdata->time.minute, navdata->time.second, navdata->time.ms);
    // printf("NAVDATA.LAT = %.6f\n", navdata->lat);
    // printf("NAVDATA.LON = %.6f\n", navdata->lon);
    // printf("NAVDATA.ALT = %.2f\n", navdata->alt);
    // printf("NAVDATA.HEADING = %.2f\n", navdata->heading);
    // printf("NAVDATA.SPEED = %.2f\n", navdata->speed);
    // printf("NAVDATA.HDOP = %.1f\n", navdata->hdop);
    // printf("NAVDATA.VDOP = %.1f\n", navdata->vdop);
    // printf("NAVDATA.PDOP = %.1f\n", navdata->pdop);
    // printf("NAVDATA.NUM_SV_FIX = %d\n", navdata->sv_inuse);
    // printf("NAVDATA.NUM_SV_VIEW = %d\n", navdata->sv_inview);

    pthread_mutex_lock(&g_lock);
    osd_data.latitude = navdata->lat;
    osd_data.longitude = navdata->lon;
    osd_data.height = navdata->alt;
    if (navdata->lon == 0 || navdata->lat == 0) {
        osd_data.GPS_status = 0;
    } else {
        osd_data.GPS_status = 1;
    }
    pthread_mutex_unlock(&g_lock);
    sem_post(&g_data_ready); // 释放信号量通知显示线程
}

int e_GPS_measure_uart_set(int gps_fd)
{
    struct termios options;

    if(tcgetattr(gps_fd,&options) < 0) {
        printf("tcgetattr error\n");
        return -1;
    }
    //设置波特率
    cfsetispeed(&options,B9600);
    cfsetospeed(&options,B9600);

    //关闭流控
    options.c_iflag &= ~(IXON | IXOFF | IXANY);  // 禁用软件流控
    options.c_cflag &= ~CRTSCTS;  // 禁用硬件流控

    //设置校验位
    options.c_cflag &= ~PARENB;  // 清除PARENB位，禁用奇偶校验
    options.c_cflag &= ~PARODD;  // 清除PARODD位（可选，无校验模式下无意义）
    options.c_cflag &= ~INPCK;   // 禁用输入奇偶校验检查

    // 设置数据位和停止位
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;       // 8数据位
    options.c_cflag &= ~CSTOPB;   // 1停止位

    // 禁用规范模式（重要！）
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

    // 设置超时和最小字符数（根据需求调整）
    options.c_cc[VMIN] = 0;       // 不等待字符
    options.c_cc[VTIME] = 10;     // 1秒超时（10分秒）

    // 禁用输出处理（可选，根据需求调整）
    options.c_oflag &= ~OPOST;
    
    if(tcsetattr(gps_fd,TCSANOW,&options) < 0) {
        printf("e GPS UART distance tcsetattr failed\n");
        return -1;
    }
    return 0;
}

// 串口读取线程函数
void *e_GPS_uart_read_method(void *arg) {

    char buffer[GPS_INFO_BUFFER_SIZE];
    int bytes_read;

    printf("e GPS ART read thread started\n");
    
    while (e_GPS_running) {

        // 读取串口数据
        bytes_read = read(e_GPS_fd, buffer, GPS_INFO_BUFFER_SIZE - 1);
        
        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            // printf("UART received: %s", buffer);
            // 打印十六进制数据
            // printf("Hex: ");
            // for (int i = 0; i < bytes_read; i++) {
            //     printf("%02X ", (char)buffer[i]);
            // }
            // printf("\n");

            for (int i = 0; i < bytes_read; i++) {
                nmea_parser_putchar(parser, (char)buffer[i]);
            }
            
        }
        
        // 短暂延时，避免CPU占用过高
        usleep(10000); // 10ms
    }
    
    printf("e GPSUARTnmea_parser_putchar(parser, uart3Packet.packet[i]); read thread exiting\n");
    return NULL;
}

int Open_GPS_Measure(){
    //打开GPS定位测距串口
    char file_name[I2C_NAME_SIZE];
	if (sprintf_s(file_name, I2C_NAME_SIZE, "/dev/ttyAMA1") < 0) {
		printf("e GPS sprintf_s error!\n");
		return -1;
	}
	e_GPS_fd = open(file_name, O_RDWR|O_NOCTTY|O_NDELAY);
	if (e_GPS_fd < 0) {
		printf("e GPS Open %s error!\n", file_name);
		return -1;
	}   
	printf("e GPS open %s success.\n", file_name);
    
	// 配置UART
    if (e_GPS_measure_uart_set(e_GPS_fd) < 0) {
        close(e_GPS_fd);
		printf("e GPS set UART error!\n");
        return -1;
    }
	printf("e GPS set UART success.\n");

    // NMEA协议
    nmea_parser_init(parser);
    parser->report_nav_status = display_navdata; 

    //读数据的串口进程
    if(pthread_create(&e_GPS_uart_thread, NULL, e_GPS_uart_read_method, NULL) != 0) {
        printf("e GPSfailed to create UART read thread");
        close(e_GPS_fd);
        return -1;
    }
    return 0;
}

int Close_GPS_Measure(){
    // 清理资源
    e_GPS_running = 0;
    pthread_join(e_GPS_uart_thread, NULL);
    close(e_GPS_fd);
    printf("e GPSprogram exited successfully\n");
    return 0;
}

//=================================================================================================================================================
// distance measure
//=================================================================================================================================================

int ly_measure_uart_set(int ly_fd)
{
    struct termios options;

    if(tcgetattr(ly_fd,&options) < 0) {
        printf("tcgetattr error\n");
        return -1;
    }
    //设置波特率
    cfsetispeed(&options,B115200);
    cfsetospeed(&options,B115200);

    //关闭流控
    options.c_iflag &= ~(IXON | IXOFF | IXANY);  // 禁用软件流控
    options.c_cflag &= ~CRTSCTS;  // 禁用硬件流控

    //设置校验位
    options.c_cflag &= ~PARENB;  // 清除PARENB位，禁用奇偶校验
    options.c_cflag &= ~PARODD;  // 清除PARODD位（可选，无校验模式下无意义）
    options.c_cflag &= ~INPCK;   // 禁用输入奇偶校验检查

    // 设置数据位和停止位
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;       // 8数据位
    options.c_cflag &= ~CSTOPB;   // 1停止位

    // 禁用规范模式（重要！）
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);

    // 设置超时和最小字符数（根据需求调整）
    options.c_cc[VMIN] = 0;       // 不等待字符
    options.c_cc[VTIME] = 10;     // 1秒超时（10分秒）

    // 禁用输出处理（可选，根据需求调整）
    options.c_oflag &= ~OPOST;
    
    if(tcsetattr(ly_fd,TCSANOW,&options) < 0) {
        printf("liangyuan measure distance tcsetattr failed\n");
        return -1;
    }
    return 0;
}

/**
 * 解析测距数据，提取目标距离
 * @param buffer 接收到的字节数据
 * @param length 数据长度
 * @return 目标距离（单位：0.1米），如果解析失败返回-1
 */
long parse_ranging_data(const unsigned char *buffer, size_t length) {
    // 检查数据长度是否符合最小要求
    if (length < 14) {
        printf("Error: Data length is too short (%zu bytes)\n", length);
        return -1;
    }
    
    // 检查STX0和CMD
    if (buffer[0] != 0x55 || buffer[1] != 0x01) {
        printf("Error: Invalid STX0 or CMD (expected 0x55 0x01, got 0x%02X 0x%02X)\n", 
               buffer[0], buffer[1]);
        return -1;
    }
    
    // // 计算校验和（异或校验）
    // unsigned char checksum = 0;
    // for (int i = 0; i < length - 1; i++) {
    //     checksum ^= buffer[i];
    // }
    
    // // 验证校验和
    // if (checksum != buffer[length - 1]) {
    //     printf("Error: Checksum mismatch (calculated 0x%02X, expected 0x%02X)\n", 
    //            checksum, buffer[length - 1]);
    //     return -1;
    // }
    
    // 提取标志字节D9
    unsigned char flag = buffer[3];
    
    // 提取三个目标距离
    unsigned long distance1 = ((buffer[4] << 16) | (buffer[5] << 8) | buffer[6]) & 0xFFFFFF;
    unsigned long distance2 = ((buffer[7] << 16) | (buffer[8] << 8) | buffer[9]) & 0xFFFFFF;
    unsigned long distance3 = ((buffer[10] << 16) | (buffer[11] << 8) | buffer[12]) & 0xFFFFFF;
    
    // 优先选择第一距离，如果不存在则选择第二距离，以此类推
    if (distance1 != 0) {
        return (long)distance1;
    } else if (distance2 != 0) {
        return (long)distance2;
    } else if (distance3 != 0) {
        return (long)distance3;
    } else {
        return 0; // 三个距离都不存在
    }
}


// 串口读取线程函数
void *ly_uart_read_method(void *arg) {
    char buffer[MEASURE_DISTANCE_BUFFER_SIZE];
    int bytes_read;
    
    printf("liangyuan measure distance UART read thread started\n");
    
    while (ly_measure_running) {
        
        pthread_mutex_lock(&ly_measure_uart_mutex);
        // 读取串口数据
        bytes_read = read(ly_measure_fd, buffer, MEASURE_DISTANCE_BUFFER_SIZE - 1);
        pthread_mutex_unlock(&ly_measure_uart_mutex);
        
        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            printf("UART received: %s", buffer);
            
            // 打印十六进制数据
            printf("Hex: ");
            for (int i = 0; i < bytes_read; i++) {
                printf("%02X ", (unsigned char)buffer[i]);
            }
            printf("\n");

            long distance = parse_ranging_data(buffer, bytes_read);
    
            if (distance >= 0) {
                printf("Target distance: %ld (0.1 meters)\n", distance);
                printf("Target distance: %.1f meters\n", distance / 10.0);
            } else {
                printf("Failed to parse ranging data\n");
            }
            pthread_mutex_lock(&g_lock);
            if (distance <= 0) {
                distance = 0;
                osd_data.Distance_status = 0;
            } else {
                osd_data.Distance_status = 1;
            }
            osd_data.distance_m = distance / 10.0;
            osd_data.now_latitude = osd_data.latitude;
            osd_data.now_longitude = osd_data.longitude;
            osd_data.now_height = osd_data.height;
            osd_data.now_fAngle_x = osd_data.fAngle_x;
            osd_data.now_fAngle_y = osd_data.fAngle_y;
            osd_data.now_fAngle_z = osd_data.fAngle_z;
            pthread_mutex_unlock(&g_lock);
            sem_post(&g_data_ready); // 释放信号量通知显示线程
        }
        
        // 短暂延时，避免CPU占用过高
        usleep(10000); // 10ms
    }
    
    printf("liangyuan measure distance UART read thread exiting\n");
    return NULL;
}

int Open_Distance_Measure(){

    //打开亮源激光测距串口
    char file_name[I2C_NAME_SIZE];
	if (sprintf_s(file_name, I2C_NAME_SIZE, "/dev/ttyAMA3") < 0) {
		printf("liangyuan measure distance sprintf_s error!\n");
		return -1;
	}
	ly_measure_fd = open(file_name, O_RDWR|O_NOCTTY|O_NDELAY);
	if (ly_measure_fd < 0) {
		printf("liangyuan measure distance Open %s error!\n", file_name);
		return -1;
	}   
	printf("liangyuan measure distance open %s success.\n", file_name);
    
	// 配置UART
    if (ly_measure_uart_set(ly_measure_fd) < 0) {
        close(ly_measure_fd);
		printf("liangyuan measure distance set UART error!\n");
        return -1;
    }
	printf("liangyuan measure distance set UART success.\n");

    //读数据的串口进程
    pthread_mutex_init(&ly_measure_uart_mutex, NULL);
    if(pthread_create(&ly_measure_uart_thread, NULL, ly_uart_read_method, NULL) != 0) {
        printf("liangyuan measure distance failed to create UART read thread");
        close(ly_measure_fd);
        return -1;
    }
    return 0;
}

int Close_Distance_Measure(){
    // 清理资源
    ly_measure_running = 0;
    pthread_join(ly_measure_uart_thread, NULL);
    close(ly_measure_fd);
    printf("liangyuan measure distance program exited successfully\n");
    return 0;
}

//=================================================================================================================================================
// OSD 字符
//=================================================================================================================================================

int update_osd_text(char * new_str ,td_s32 handle_overlayex_num) {

    // 更新OSD显示
    Sample_SurfaceWord_ToBMP(new_str,&stBitmap); 
  
    ot_rgn_canvas_info canvas_info;
    td_s32 ret = ss_mpi_rgn_get_canvas_info(handle_overlayex_num, &canvas_info);
    if (ret != TD_SUCCESS) {
        sample_print("osd ss_mpi_rgn_get_canvas_info failed with %#x!\n", ret);
        return -1;
    }
    CopyBmpToCanvas(&stBitmap ,&canvas_info);
    canvas_info.size.width = stBitmap.width;
    canvas_info.size.height = stBitmap.height;
    canvas_info.pixel_format = stBitmap.pixel_format;

    ret = ss_mpi_rgn_update_canvas(handle_overlayex_num);
    if (ret != TD_SUCCESS) {
        sample_print("osd ss_mpi_rgn_update_canvas failed with %#x!\n", ret);
        return -1;
    }
    
    return 0;
}
// OSD显示线程
void* osd_display_thread(void* arg) {
    
    char display_text[512];
    td_s32 handle_overlayex_num = 21;//OVERLAYEX_MIN_HANDLE 20
    double x2, y2, z2; // 计算引导目标点

    while (1) {
        // 等待数据就绪
        sem_wait(&g_data_ready);
        
        pthread_mutex_lock(&g_lock);
        if (update_osd_required) {

            memset(display_text, 0, sizeof(display_text));
            if (osd_data.Distance_status == 0) {
                snprintf(display_text, sizeof(display_text), "等待测距");
            } else {
                if (osd_data.GPS_status == 0 || osd_data.Compass_status == 0) {
                    snprintf(display_text, sizeof(display_text), "测距%.2f米",osd_data.distance_m);
                } else {
                    calculate_target_point(osd_data.now_longitude, osd_data.now_latitude, osd_data.now_height, 
                        osd_data.now_fAngle_x, osd_data.now_fAngle_y, osd_data.now_fAngle_z, osd_data.distance_m, 
                        &x2, &y2, &z2);
                    snprintf(display_text, sizeof(display_text), "测距%.2f米(%.4f°,%.4f°,%.1fm)",
                        osd_data.distance_m, x2, y2, z2);
                }
            }
            update_osd_text(display_text ,handle_overlayex_num);

            memset(display_text, 0, sizeof(display_text));
            snprintf(display_text, sizeof(display_text), "纬度%.2f°,经度%.2f°,海拔%.1f米",osd_data.latitude, osd_data.longitude,osd_data.height);
            update_osd_text(display_text ,handle_overlayex_num+1);

            memset(display_text, 0, sizeof(display_text));
            snprintf(display_text, sizeof(display_text), "俯仰角%.1f度,横滚角%.1f度,偏航角%.1f度", osd_data.fAngle_x,osd_data.fAngle_y,osd_data.fAngle_z);
            update_osd_text(display_text ,handle_overlayex_num+2);
            
            // //复制到动态分配的字符串
            // char* new_str = strdup(display_text);
            // if (new_str == NULL) {
            //     fprintf(stderr, "内存分配失败!\n");
            //     break;
            // }
            
            // // 释放内存
            // free(new_str);
            // new_str = NULL;
        }
        pthread_mutex_unlock(&g_lock);

        usleep(500000);
    }
    return NULL;
}

//====================================================================================================
//  media stream
//====================================================================================================

static td_s32 sample_vio_start_venc(sample_sns_type sns_type,ot_venc_chn venc_chn[], size_t size, td_u32 chn_num)
{
    printf("shikeDebug  vio run  start venc\n");

    td_s32 i;
    td_s32 ret;

    if (chn_num > size) {
        return TD_FAILURE;
    }

    // shikeDebug
    //sample_comm_vi_get_size_by_sns_type(sns_type, &g_venc_chn_param.venc_size);
    
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

static td_s32 sample_vio_start_venc_and_vo(sample_sns_type sns_type,ot_vpss_grp vpss_grp[], size_t size, td_u32 grp_num)
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

    ret = sample_vio_start_venc(sns_type,venc_chn, sizeof(venc_chn) / sizeof(venc_chn[0]), grp_num);
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

static td_s32 sample_vio_start_route(sample_sns_type sns_type,sample_vi_cfg *vi_cfg, sample_vpss_cfg *vpss_cfg, td_s32 route_num)
{
    printf("shikeDebug  vio run  start route have grp\n");

    td_s32 i, j, ret;
    ot_vpss_grp vpss_grp[SAMPLE_VIO_MAX_ROUTE_NUM] = {0, 1, 2, 3};

    sample_comm_vi_get_size_by_sns_type(sns_type, &g_vb_param.vb_size);
    if (sample_vio_sys_init() != TD_SUCCESS) {
        return TD_FAILURE;
    }

    for (i = 0; i < route_num; i++) {
        ret = sample_comm_vi_start_vi(sns_type,&vi_cfg[i]);
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
        ret = sample_vio_start_vpss(sns_type,vpss_grp[i], vpss_cfg);
        if (ret != TD_SUCCESS) {
            goto start_vpss_failed;
        }
    }

    ret = sample_vio_start_venc_and_vo(sns_type,vpss_grp, SAMPLE_VIO_MAX_ROUTE_NUM, route_num);
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

static td_s32 sample_vio_all_mode(sample_sns_type sns_type)
{
    printf("shikeDebug  vio run all mode.\n");

    sample_vi_cfg vi_cfg[1];
    sample_vpss_cfg vpss_cfg;

    g_vio_sys_cfg.mode_type = OT_VI_OFFLINE_VPSS_OFFLINE;
    g_vio_sys_cfg.vi_fmu[0] = OT_FMU_MODE_OFF;
    g_vb_param.blk_num[0] =  6; /* raw_vb num 6 or 3 */

    sample_comm_vi_get_vi_cfg_by_fmu_mode(sns_type, g_vio_sys_cfg.vi_fmu[0], &vi_cfg[0]);
    sample_comm_vpss_get_default_vpss_cfg(sns_type, &vpss_cfg, g_vio_sys_cfg.vpss_fmu[0]);

    if (sample_vio_start_route(sns_type,vi_cfg, &vpss_cfg, g_vio_sys_cfg.route_num) != TD_SUCCESS) {
        return TD_FAILURE;
    }

    return TD_SUCCESS;
}

static td_s32 sample_vio_all_mode_stop(sample_sns_type sns_type)
{
    printf("shikeDebug  vio run all mode.\n");
    
    // sample_destroy_region();

    sample_vi_cfg vi_cfg[1];
    sample_vpss_cfg vpss_cfg;

    g_vio_sys_cfg.mode_type = OT_VI_OFFLINE_VPSS_OFFLINE;
    g_vio_sys_cfg.vi_fmu[0] = OT_FMU_MODE_OFF;
    g_vb_param.blk_num[0] =  6; /* raw_vb num 6 or 3 */

    sample_comm_vi_get_vi_cfg_by_fmu_mode(sns_type, g_vio_sys_cfg.vi_fmu[0], &vi_cfg[0]);
    sample_comm_vpss_get_default_vpss_cfg(sns_type, &vpss_cfg, g_vio_sys_cfg.vpss_fmu[0]);

    sample_vio_stop_route(vi_cfg, g_vio_sys_cfg.route_num);

    sample_destroy_region();

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

td_s32 sample_region_do_destroy(td_s32 handle_num, ot_rgn_type type, ot_mpp_chn *chn, region_op_flag flag)
{
    td_s32 ret = TD_SUCCESS;
    td_s32 i;
    td_s32 min_handle;
    td_s32 handle_overlayex_num = 21;

    if (handle_num <= 0 || handle_num > 16) { /* 16:max region num */
        sample_print("handle_num is illegal %d!\n", handle_num);
        return TD_FAILURE;
    }
    if (type < 0 || type >= OT_RGN_BUTT) {
        sample_print("type is illegal %d!\n", type);
        return TD_FAILURE;
    }
    if (chn == TD_NULL) {
        sample_print("mpp_chn is NULL !\n");
        return TD_FAILURE;
    }

    if (flag & REGION_OP_CHN) {
        ret = ss_mpi_rgn_detach_from_chn(handle_overlayex_num, &chn);
        if (ret != TD_SUCCESS) {
            sample_print("ss_mpi_rgn_detach_from_chn failed with %#x!\n", ret);
            return TD_FAILURE;
        }
    }

    if (flag & REGION_DESTROY) {
        ret = ss_mpi_rgn_destroy(handle_overlayex_num);
        if (ret != TD_SUCCESS) {
            sample_print("ss_mpi_rgn_destroy failed with %#x!\n", ret);
            return TD_FAILURE;
        }
    }
    return ret;
}


td_s32 sample_destroy_region(td_void){
    printf("region do destroy ...\n");
    ot_rgn_type type;
    ot_mpp_chn chn;

    td_s32 handle_start_num = RGN_HANDLE_NUM_8;
    type = OT_RGN_OVERLAYEX;
    chn.mod_id = OT_ID_VPSS;
    chn.dev_id = 0;
    chn.chn_id = 0;
    sample_region_do_destroy(handle_start_num, type, &chn, REGION_OP_CHN | REGION_DESTROY);

    return 0;
}

td_s32 sample_region_vpss_overlayex(td_void)
{
    printf("region vpss overlayex ...\n");

    OpenOsdText();
    Sample_SurfaceWord_ToBMP("**##**",&stBitmap); 

    ot_rgn_type type;
    ot_mpp_chn chn;
    type = OT_RGN_OVERLAYEX;
    chn.mod_id = OT_ID_VPSS;
    chn.dev_id = 0;
    chn.chn_id = 0;

    td_s32 i, ret, min_handle;
    td_s32 handle_start_num = RGN_HANDLE_NUM_8;
    region_op_flag op_flag = REGION_OP_CHN;

    rgn_check_handle_num_return(handle_start_num);
    if (handle_start_num <= 0 || handle_start_num > 16) { /* 16:max_num */
        sample_print("handle_num is illegal %d!\n", handle_start_num);
        return TD_FAILURE;
    }
    if (type < 0 || type >= OT_RGN_BUTT) {
        sample_print("type is illegal %d!\n", type);
        return TD_FAILURE;
    }

    // create region
    ot_rgn_attr region;
    region.type = OT_RGN_OVERLAYEX;
    region.attr.overlayex.pixel_format = OT_PIXEL_FORMAT_ARGB_1555;
    region.attr.overlayex.size.height = MAX_VALUE(stBitmap.height,160);
    region.attr.overlayex.size.width = 1010; //RGN_DEFAULT_HEIGHT 160
    //region.attr.overlayex.bg_color = 0x00ff;
    region.attr.overlayex.bg_color = 0x0000;
    region.attr.overlayex.canvas_num = 2;//DEFAULT_CANVAS_NUM 2
    td_s32 handle_overlayex_num = 21;//OVERLAYEX_MIN_HANDLE 20
    for (i = handle_overlayex_num; i <handle_overlayex_num + 3; i++) {
        ret = ss_mpi_rgn_create(i, &region);
        if (ret != TD_SUCCESS) {
            sample_print("ss_mpi_rgn_create failed with %#x!\n", ret);
            return TD_FAILURE;
        }
    }
    
    // attach region
    ot_rgn_chn_attr chn_attr;

    for (i = handle_overlayex_num; i <handle_overlayex_num + 3; i++) {
        chn_attr.is_show = TD_TRUE;
        chn_attr.type = OT_RGN_OVERLAYEX;
        chn_attr.attr.overlayex_chn.bg_alpha = 0;//RGN_ALPHA 128
        chn_attr.attr.overlayex_chn.fg_alpha = 255;
        chn_attr.attr.overlayex_chn.point.x = 10;
        chn_attr.attr.overlayex_chn.point.y = ALIGN_2(50 * (i - 20));
        chn_attr.attr.overlayex_chn.layer = handle_overlayex_num - 20;

        if (op_flag & REGION_OP_CHN) {
            ret = ss_mpi_rgn_attach_to_chn(i, &chn, &chn_attr);
            if (ret != TD_SUCCESS) {
                sample_print("ss_mpi_rgn_attach_to_chn failed with %#x!\n", ret);
                return TD_FAILURE;
            }
        } else if (op_flag & REGION_OP_DEV) {
            ret = ss_mpi_rgn_attach_to_dev(i, &chn, &chn_attr);
            if (ret != TD_SUCCESS) {
                sample_print("ss_mpi_rgn_attach_to_dev failed with %#x!\n", ret);
                return TD_FAILURE;
            }
        }
    }
    min_handle = sample_comm_region_get_min_handle(type);
    if (sample_comm_check_min(min_handle) != TD_SUCCESS) {
        sample_print("min_handle(%d) should be in [0, %d).\n", min_handle, OT_RGN_HANDLE_MAX);
        return -1;
    }
    for (i = handle_overlayex_num; i <handle_overlayex_num + 3; i++) {
        printf("shikeDebug sample_comm_region_get_up_canvas %d\n",i);
        if (i == handle_overlayex_num) {
            Sample_SurfaceWord_ToBMP("等待测距",&stBitmap);            
        } else if (i == handle_overlayex_num+1) {
            Sample_SurfaceWord_ToBMP("纬度0,经度0,海拔0",&stBitmap); 
        } else if (i == handle_overlayex_num+2) {
            Sample_SurfaceWord_ToBMP("俯仰角0度,横滚角0度,偏航角0度",&stBitmap);
        }
        ot_rgn_canvas_info canvas_info;
        ret = ss_mpi_rgn_get_canvas_info(i, &canvas_info);
        if (ret != TD_SUCCESS) {
            sample_print("ss_mpi_rgn_get_canvas_info failed with %#x!\n", ret);
            return TD_FAILURE;
        }
        CopyBmpToCanvas(&stBitmap ,&canvas_info);
        canvas_info.size.width = stBitmap.width;
        canvas_info.size.height = stBitmap.height;
        canvas_info.pixel_format = stBitmap.pixel_format;

        ret = ss_mpi_rgn_update_canvas(i);
        if (ret != TD_SUCCESS) {
            sample_print("ss_mpi_rgn_update_canvas failed with %#x!\n", ret);
            return TD_FAILURE;
        }
    }
    if (ret != TD_SUCCESS) {
        sample_print("sample_comm_region_get_up_canvas failed!\n");
        return -1;
    }

    return 0;
}

void app_start(sample_sns_type sns_type)
{
    sample_register_sig_handler(sample_vio_handle_sig);

    if (sample_ipc_server_init(sample_vio_ipc_msg_proc) != TD_SUCCESS) {
        printf("sample_ipc_server_init failed!!!\n");
    }
    
    td_s32 ret = sample_vio_all_mode(sns_type);
    if ((ret == TD_SUCCESS) && (g_sig_flag == 0)) {
        printf("\033[0;32mprogram exit normally!\033[0;39m\n");
    } else {
        printf("\033[0;31mprogram exit abnormally!\033[0;39m\n");
    }

    sample_region_vpss_overlayex();

}


int Open_Infrared_Vio(){
    printf("open infrared vio ...\n");
    
    //open infrared 
    app_start(FPGA_BT1120_14BIT);
    usleep(1000);
    update_osd_required = true;
    printf("open infrared done\n");
    return 0;
}

int Close_Infrared_Vio(){
    printf("close infrared vio ...\n");
    update_osd_required = false;
    sample_vio_all_mode_stop(FPGA_BT1120_14BIT);

    usleep(100*1000);
    sample_ipc_server_deinit();
    usleep(1000);
    printf("close infrared vio done\n");
    return 0;
}

int Open_Colorcamera_Vio(){
    printf("open colorcamera vio ...\n");
    
    //open colorcamera 
    app_start(COLORCAMERA_MIPIRX_YUV422);
    usleep(1000);
    update_osd_required = true;
    printf("open colorcamera vio done\n");
    return 0;
}

int Close_Colorcamera_Vio(){
    printf("close colorcamera vio ...\n");
    update_osd_required = false;
    sample_vio_all_mode_stop(COLORCAMERA_MIPIRX_YUV422);

    usleep(100*1000);
    sample_ipc_server_deinit();
    usleep(1000);
    printf("close colorcamera vio done\n");
    return 0;
}

// 定义系统状态
typedef enum {
    STATE_NO,
    STATE_A,  // 打开红外
    STATE_B   // 打开可见光
} SystemState;

// 共享资源结构体
typedef struct {
    pthread_mutex_t mutex;      // 互斥锁
    pthread_cond_t cond;        // 条件变量
    int button_id;              // 按钮编号
    SystemState state;          // 当前系统状态
    int exit_flag;              // 退出标志
} ButtonEvent;

#define MAX_EVENTS 16
ButtonEvent g_event;            // 全局事件对象
static int button_fd = -1;      // 按键设备文件描述符
// 环形缓冲区存储按键事件
struct {
    unsigned char events[MAX_EVENTS];
    int head;
    int tail;
    int count;
    pthread_mutex_t lock;
    pthread_cond_t cond;
} event_buffer = {0};

// 信号处理函数
void signal_handler(int signum) {
    if (signum == SIGIO) {
        unsigned char button_state;
        
        // 读取按键状态（非阻塞）
        int flags = fcntl(button_fd, F_GETFL);
        fcntl(button_fd, F_SETFL, flags | O_NONBLOCK);
        
        if (read(button_fd, &button_state, 1) == 1) {
            // 恢复阻塞模式
            fcntl(button_fd, F_SETFL, flags);
            
            // 加锁保护共享资源
            pthread_mutex_lock(&event_buffer.lock);
            
            // 如果缓冲区未满，添加新事件
            if (event_buffer.count < MAX_EVENTS) {
                event_buffer.events[event_buffer.head] = button_state;
                event_buffer.head = (event_buffer.head + 1) % MAX_EVENTS;
                event_buffer.count++;
                
                printf("Button event added: %d (buffer count: %d)\n", 
                       button_state, event_buffer.count);
                
                // 通知主线程有新事件
                pthread_cond_signal(&event_buffer.cond);
            } else {
                printf("Button event buffer full, dropping event: %d\n", button_state);
            }
            
            // 解锁
            pthread_mutex_unlock(&event_buffer.lock);
            
            // 检查退出标志
            if (!g_event.exit_flag && button_state != 0) {
                
                printf("button read : %d\n",button_state);

                // 更新按钮编号
                g_event.button_id = button_state;
                
                // 通知主线程
                pthread_cond_signal(&g_event.cond);
            }
            
            // 解锁
            pthread_mutex_unlock(&g_event.mutex);
        }
    }
}

// 子线程函数：监控按键
void* button_monitor_thread(void* arg) {
    // 打开按键设备
    button_fd = open("/dev/buttons", O_RDWR);
    if (button_fd < 0) {
        perror("Failed to open buttons device");
        return NULL;
    }
    
    // 设置异步I/O
    fcntl(button_fd, F_SETOWN, getpid());
    int flags = fcntl(button_fd, F_GETFL);
    fcntl(button_fd, F_SETFL, flags | FASYNC);
    
    // 设置信号处理
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sa.sa_flags = SA_RESTART;  // 自动重启被信号中断的系统调用
    sigaction(SIGIO, &sa, NULL);
    
    // 等待退出标志
    pthread_mutex_lock(&g_event.mutex);
    while (!g_event.exit_flag) {
        pthread_cond_wait(&g_event.cond, &g_event.mutex);
    }
    pthread_mutex_unlock(&g_event.mutex);
    
    // 清理资源
    close(button_fd);

    pthread_mutex_destroy(&event_buffer.lock);
    pthread_cond_destroy(&event_buffer.cond);

    return NULL;
}

int main() {
    // 初始化互斥锁和条件变量
    pthread_mutex_init(&g_event.mutex, NULL);
    pthread_cond_init(&g_event.cond, NULL);
    g_event.button_id = 0;
    g_event.exit_flag = 0;

    gpio_test_out(3,4,0);    //选择可见光(0)/红外(1)的串口
    // 可见光相机的状态初始化
    send_uart_commond(colorcamera_set_1080p ,0);
    send_uart_commond(colorcamera_wide_engle ,0);

    // 默认执行函数A
    g_event.state = STATE_A;    
    Open_Infrared_Vio();

    gpio_test_out(3,4,1);    //选择可见光(0)/红外(1)的串口

    gpio_test_out(3,0,1); //distance measure
    gpio_test_out(3,1,1); //GPS measure
    gpio_test_out(3,2,1); //compass measure

    // OSD字符,初始化信号量和互斥锁
    sem_init(&g_data_ready, 0, 0);
    pthread_mutex_init(&g_lock, NULL);
    update_osd_required = false;

    //
    Open_Compass_Measure();

    //
    Open_GPS_Measure();

    //
    Open_Distance_Measure();

     // 创建三个线程
    if (pthread_create(&osd_tid, NULL, osd_display_thread, NULL) != 0) {
        printf("Failed to create osd threads!\n");
        return -1;
    }

    // 创建子线程
    pthread_t monitor_thread;
    if (pthread_create(&monitor_thread, NULL, button_monitor_thread, NULL) != 0) {
        perror("Failed to create button monitor thread");
    }

    if (init_colorcamera_controller() != 0) {
        perror("Failed to create colorcamera uart monitor thread");
    }

    update_osd_required = true;

    // 主线程：处理按钮事件
    while (1) {
        unsigned char button_state = 0;
        
        // 加锁等待事件
        pthread_mutex_lock(&event_buffer.lock);
        
        // 如果缓冲区为空且未退出，则等待
        while (event_buffer.count == 0 && !g_event.exit_flag) {
            pthread_cond_wait(&event_buffer.cond, &event_buffer.lock);
        }
        
        // 检查是否需要退出
        if (g_event.exit_flag) {
            pthread_mutex_unlock(&event_buffer.lock);
            break;
        }
        
        // 从缓冲区取出事件
        button_state = event_buffer.events[event_buffer.tail];
        event_buffer.tail = (event_buffer.tail + 1) % MAX_EVENTS;
        event_buffer.count--;

        printf("Main thread: Processing button %d (buffer count: %d)\n", 
               button_state, event_buffer.count);
        
        // 解锁
        pthread_mutex_unlock(&event_buffer.lock);

        // 复制参数字符串用于处理
        unsigned  char hex_str[256];
        
        // 执行按钮对应的操作
        switch (button_state)
        {
            //红外视频/可见光视频
            case 2:
                // 根据当前状态执行相应函数
                if (g_event.state == STATE_A) {
                    // 释放互斥锁，允许按钮线程在执行函数A时更新状态
                    g_event.state = STATE_B;
                    send_uart_commond(infrared_zoom_stop ,1);
                    Close_Infrared_Vio();
                    Open_Colorcamera_Vio();
                    gpio_test_out(3,4,0);    //选择可见光(0)/红外(1)的串口
                } else if (g_event.state == STATE_B) {
                    g_event.state = STATE_A;
                    set_zoom_state(ZOOM_DECREASE);
                    Close_Colorcamera_Vio();
                    Open_Infrared_Vio();
                    gpio_test_out(3,4,1);    //选择可见光(0)/红外(1)的串口
                } else {
                }
                break;

            //变倍&变焦+  按键按下
            case 5:
                printf("+++ 111 ...\n");
                if (g_event.state == STATE_A) {
                    // 红外
                    send_uart_commond(infrared_zoom_in ,1);
                    printf("Infrared Zooming in...\n");
                }else if (g_event.state == STATE_B) {
                    // 可见光
                    set_zoom_state(ZOOM_INCREASE);
                    printf("Colorcamera Zooming in...\n");
                } else {

                }
                break;

            //变倍&变焦+  按键抬起
            case 6:
                printf("+++ 222...\n");
                if (g_event.state == STATE_A) {
                    // 红外
                    send_uart_commond(infrared_zoom_stop ,1);
                    send_uart_commond(infrared_auto_focus_now ,1);
                    printf("Infrared Zooming stop.\n");
                }else if (g_event.state == STATE_B) {
                    // 可见光
                    set_zoom_state(ZOOM_STOP);
                    printf("Colorcamera Zoom stopped at %.1fx\n", get_current_magnification());
                } else {

                }
                
                break;

            //变倍&变焦-  按键按下
            case 9:
                printf("--- 111...\n");
                if (g_event.state == STATE_A) {
                    // 红外
                    send_uart_commond(infrared_zoom_out ,1);
                    printf("Infrared Zooming out...\n");
                }else if (g_event.state == STATE_B) {
                    // 可见光
                    set_zoom_state(ZOOM_DECREASE);
                    printf("Colorcamera Zooming out...\n");
                } else {

                }
                
                break;

            //变倍&变焦-  按键抬起
            case 10:
                printf("--- 222...\n");
                if (g_event.state == STATE_A) {
                    // 红外
                    send_uart_commond(infrared_zoom_stop ,1);
                    send_uart_commond(infrared_auto_focus_now ,1);
                    printf("Infrared Zooming stop.\n");
                }else if (g_event.state == STATE_B) {
                    // 可见光
                    set_zoom_state(ZOOM_STOP);
                    printf("Colorcamera Zoom stopped at %.1fx\n", get_current_magnification());    
                } else {

                }
                
                break;

            // 背景矫正
            case 13:
                printf("+++ 13...\n");
                send_uart_commond(infrared_set_background ,1);
                printf("Background correcting.\n");
                break;

            // 激光测距
            case 16:
                 strncpy(hex_str, "55 01 02 00 00 56\n", sizeof(hex_str) - 1);
                 hex_str[sizeof(hex_str) - 1] = '\0';
                 
                 // 移除空格
                 remove_spaces(hex_str);
                 printf("Processed hex string: %s\n", hex_str);
                 
                 // 转换为二进制数据
                 unsigned char bin_data[128];
                 int bin_len = hex_string_to_bytes(hex_str, bin_data, sizeof(bin_data));
                 if (bin_len < 0) {
                     printf("Processed hex to string failure.\n");
                     break;
                 }
                 
                 // 打印转换后的二进制数据
                 printf("Binary data  %d  to send: ",bin_len);
                 for (int i = 0; i < bin_len; i++) {
                     printf("%02X ", bin_data[i]);
                 }
                 printf("\nbinary: ");
                 for (int i = 0; i < bin_len; i++) {
                     print_binary(bin_data[i]);
                 }
                 printf("\n");
 
                 // 加锁保护串口写入
                 pthread_mutex_lock(&ly_measure_uart_mutex);
                 int bytes_written = write(ly_measure_fd, bin_data, bin_len);
                 pthread_mutex_unlock(&ly_measure_uart_mutex);
 
                 if (bytes_written < 0) {
                     perror("liangyuan measure distance UART write error");
                 } else {
                     printf("liangyuan measure distance sent %d bytes to UART\n", bytes_written);
                 }
                break;
            
            default:
                break;
        }
    }

    // 通知子线程退出并等待
    pthread_mutex_lock(&g_event.mutex);
    g_event.exit_flag = 1;
    pthread_cond_signal(&g_event.cond);
    pthread_mutex_unlock(&g_event.mutex);
    pthread_join(monitor_thread, NULL);

    // 清理资源
    pthread_mutex_destroy(&g_event.mutex);
    pthread_cond_destroy(&g_event.cond);

    return 0;
}
