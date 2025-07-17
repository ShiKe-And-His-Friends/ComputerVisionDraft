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



int main(int argc, char **argv) {
    int file;
    char *bus = "/dev/i2c-6";  // 指定I2C总线
    int addr;
    unsigned char data;

    // 打开I2C总线
    if ((file = open(bus, O_RDWR)) < 0) {
        perror("Failed to open the bus");
        return 1;
    }

    printf("     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f\n");
    for (int i = 0; i < 128; i += 16) {
        printf("%02x: ", i);
        for (int j = 0; j < 16; j++) {
            addr = i + j;
            if (addr < 0x03 || addr > 0x77) {
                printf("   ");
                continue;
            }

            if (ioctl(file, I2C_SLAVE, addr) < 0) {
                printf("-- ");
                continue;
            }

            if (write(file, &data, 0) < 0) {
                printf("-- ");
                continue;
            }

            printf("%02x ", addr);
        }
        printf("\n");
    }

    close(file);
    return 0;
}