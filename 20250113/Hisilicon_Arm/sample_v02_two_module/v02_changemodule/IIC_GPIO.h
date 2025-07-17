#ifndef IIC_GPIO_H
#define IIC_GPIO_H

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

int gpio_test_out(unsigned int gpio_chip_num, unsigned int gpio_offset_num,unsigned int gpio_out_val);

#endif