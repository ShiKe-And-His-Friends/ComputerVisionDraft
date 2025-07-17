
#ifndef UART_CAMERA_H
#define UART_CAMERA_H

/*
  Copyright (c), 2001-2024, Shenshu Tech. Co., Ltd.
 */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <stdio.h>
#include <ctype.h>

#include <stdio.h>
#include <string.h>

#include "ot_osal.h"

#include "sample_comm.h"

//I2C include 
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <ctype.h>

#include <sys/mman.h>
#include <memory.h>
#include <linux/fb.h>
#include "securec.h"


#include <linux/module.h>
#include <linux/kernel.h>
#include "ot_osal.h"
#include "ot_type.h"
#include "ot_common.h"
#include "ot_defines.h"
#include "osal_ioctl.h"

#include<termios.h>

#define I2C_NAME_SIZE 0X80

static char *infrared_set_background[3]   = {"2", "55 AA 06 10 05 00 00 00 00 15 FF", NULL};
static char *infrared_zoom_in[3]   = {"2", "55 AA 06 11 16 00 00 00 00 27 FF", NULL};
static char *infrared_zoom_out[3]   = {"2", "55 AA 06 11 15 00 00 00 00 26 FF", NULL};
static char *infrared_zoom_stop[3]   = {"2", "55 AA 06 11 19 00 00 00 00 2A FF", NULL};
static char *infrared_auto_focus_now[3]   = {"2", "55 AA 06 11 1E 00 00 00 00 2F FF", NULL};

static char *colorcamera_set_1080p[3]   = {"2", "81 01 04 24 73 00 07 FF", NULL};
static char *colorcamera_wide_engle[3]  = {"2", "81 01 04 07 03 FF", NULL};
static char *colorcamera_long_focus[3]  = {"2", "81 01 04 07 02 FF", NULL};

/**************************************************************
 * 执行uart串口指令
 * char *argv[3] = {"2", "81 01 04 48 01 00 00 00 FF", NULL};
 * int result = send_uart_commond(argv, 0);
 * @param setiospeed 0:9600 1:115200
 * ***********************************************************/
int send_uart_commond(td_char *argv[] ,int setiospeed);

#endif