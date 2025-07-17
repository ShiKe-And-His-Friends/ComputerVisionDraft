#ifndef _INCLUDE_H
#define _INCLUDE_H

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
#include "bsp_type.h"
#include "i2c_dev.h"
#include "strfunc.h"

#include <linux/module.h>
#include <linux/kernel.h>
#include "ot_osal.h"
#include "ot_type.h"
#include "ot_common.h"
#include "ot_defines.h"
#include "osal_ioctl.h"

#include "Common.h"
#include "LontiumDrv.h"

#include "LT9611.h"
#include "OcmI2cMaster.h"
#include "LontiumDrv.h"


extern u8 HDMI_WriteI2C_Byte(u16 addr,u16 data);
extern u8 HDMI_ReadI2C_Byte(u16 addr);

#endif
