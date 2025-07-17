/*
  Copyright (c), 2001-2024, Shenshu Tech. Co., Ltd.
 */

//Hisi_Offical Self_Devp
#define Hisi_Offical 0
#define Self_Devp 1
#define ARM_BOARD_TYPE Self_Devp

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

// #ifdef __cplusplus
// #if __cplusplus
// extern "C" {
// #endif
// #endif /* __cplusplus */

#define READ_MIN_CNT 5
#define READ_MAX_CNT 8
#define WRITE_MIN_CNT 5
#define WRITE_MAX_CNT 7
#define INT_MAX 2147483647
#define I2C_NAME_SIZE 0X10

#define I2C_READ_STATUS_OK 2
#define I2C_MSG_CNT 2
#define BYTE_CNT_2 2
#define BIT_CNT_8 8

struct i2c_rdwr_args {
	unsigned int i2c_num;
	unsigned int dev_addr;
	unsigned int reg_addr;
	union {
		unsigned int data;
		unsigned int reg_addr_end;
	} w_r_union;
	unsigned int reg_width;
	unsigned int data_width;
	unsigned int reg_step;
};

#define I2C_READ_BUF_LEN 4

static int i2c_open(int *fd, struct i2c_rdwr_args args)
{
	char file_name[I2C_NAME_SIZE];
	if (sprintf_s(file_name, I2C_NAME_SIZE, "/dev/i2c-%u", args.i2c_num) < 0) {
		printf("sprintf_s error!\n");
		return -1;
	}

	*fd = open(file_name, O_RDWR);
	if (*fd < 0) {
		printf("Open %s error!\n", file_name);
		return -1;
	}
	return 0;
}

static int i2c_set_mode(int fd,  struct i2c_rdwr_args args)
{
	if (fd < 0)
		return -1;
	if (ioctl(fd, I2C_SLAVE_FORCE, args.dev_addr)) {
		printf("set i2c device address error!\n");
		return -1;
	}
	return 0;
}

static int i2c_ioc_init(struct i2c_rdwr_ioctl_data *rdwr,
			unsigned char *buf,
			size_t buf_size,
			struct i2c_rdwr_args args)
{
	if (memset_s(buf, buf_size, 0, I2C_READ_BUF_LEN) != EOK) {
		printf("memset_s fail!\n");
		return -1;
	}
	rdwr->msgs[0].addr = args.dev_addr;
	rdwr->msgs[0].flags = 0;
	rdwr->msgs[0].len = args.reg_width;
	rdwr->msgs[0].buf = buf;

	rdwr->msgs[1].addr = args.dev_addr;
	rdwr->msgs[1].flags = 0;
	rdwr->msgs[1].flags |= I2C_M_RD;
	rdwr->msgs[1].len = args.data_width;
	rdwr->msgs[1].buf = buf;

	return 0;
}

static int i2c_read_regs(int fd, struct i2c_rdwr_args args)
{
	unsigned int cur_addr;
	unsigned int data;
	unsigned char buf[I2C_READ_BUF_LEN];
	static struct i2c_rdwr_ioctl_data rdwr;
	static struct i2c_msg msg[I2C_MSG_CNT];

	rdwr.msgs = &msg[0];
	rdwr.nmsgs = (__u32)I2C_MSG_CNT;
	if (i2c_ioc_init(&rdwr, buf, sizeof(buf), args) != 0)
		return -1;

	for (cur_addr = args.reg_addr; cur_addr <= args.w_r_union.reg_addr_end; cur_addr += args.reg_step) {
		if (args.reg_width == BYTE_CNT_2) {
			buf[0] = (cur_addr >> BIT_CNT_8) & 0xff;
			buf[1] = cur_addr & 0xff;
		} else {
			buf[0] = cur_addr & 0xff;
		}

		if (ioctl(fd, I2C_RDWR, &rdwr) != I2C_READ_STATUS_OK) {
			printf("CMD_I2C_READ error!\n");
			return -1;
		}

		if (args.data_width == BYTE_CNT_2)
			data = buf[1] | (buf[0] << BIT_CNT_8);
		else
			data = buf[0];

		printf("0x%x: 0x%x\n", cur_addr, data);
	}

	return 0;
}

static int i2c_write_reg(int fd, struct i2c_rdwr_args args)
{
	unsigned char buf[4];
	int index = 0;

	if (args.reg_width == BYTE_CNT_2) {
		buf[index] = (args.reg_addr >> BIT_CNT_8) & 0xff;
		index++;
		buf[index] = args.reg_addr & 0xff;
		index++;
	} else {
		buf[index] = args.reg_addr & 0xff;
		index++;
	}

	if (args.data_width == BYTE_CNT_2) {
		buf[index] = (args.w_r_union.data >> BIT_CNT_8) & 0xff;
		index++;
		buf[index] = args.w_r_union.data & 0xff;
	} else {
		buf[index] = args.w_r_union.data & 0xff;
	}

	if (write(fd, buf, (args.reg_width + args.data_width)) < 0) {
		printf("i2c write error!\n");
		return -1;
	}

	return 0;
}

int i2c_read(int argc, char *argv[])
{
	int retval = 0;
	int fd = -1;
	struct i2c_rdwr_args args = {
		.i2c_num = 0,
		.dev_addr = 0,
		.reg_addr = 0,
		.w_r_union.reg_addr_end = 0,
		.reg_width = 2,
		.data_width = 1,
		.reg_step = 1
	};

	// if (parse_args(argc, argv, &args) != 0) {
	// 	print_r_usage();
	// 	return -1;
	// }

	printf("i2c_num:0x%x, dev_addr:0x%x; reg_addr:0x%x; reg_addr_end:0x%x;"
		"reg_width: %u; data_width: %u; reg_step: %u. \n",
		args.i2c_num, args.dev_addr , args.reg_addr, args.w_r_union.reg_addr_end,
		args.reg_width, args.data_width, args.reg_step);

	if (i2c_open(&fd, args) != 0)
		return -1;

	if (i2c_set_mode(fd, args) != 0) {
		retval = -1;
		goto end;
	}

	if (i2c_read_regs(fd, args) != 0) {
		retval = -1;
		goto end;
	}

	retval = 0;
end:
	close(fd);
	return retval;
}

// usage: i2c_write <i2c_num> <dev_addr> <reg_addr> <value> <reg_width> <data_width>. sample:
// 0x1 0x56 0x0 0x28 2 2
// 0x1 0x56 0x0 0x28. default reg_width and data_width is 1.
int i2c_write(int argc, char *argv[])
{
	int retval = 0;
	int fd = -1;

	struct i2c_rdwr_args args = {
		.i2c_num = 0,
		.dev_addr = 0,
		.reg_addr = 0,
		.w_r_union.data = 0,
		.reg_width = 2,
		.data_width = 1,
	};

	// if (parse_args(argc, argv, &args) != 0) {
	// 	print_w_usage();
	// 	return -1;
	// }

	printf("i2c_num:0x%x, dev_addr:0x%x; reg_addr:0x%x; data:0x%x;"
		"reg_width: %u; data_width: %u.\n",
		args.i2c_num, args.dev_addr , args.reg_addr, args.w_r_union.data,
		args.reg_width, args.data_width);

	if (i2c_open(&fd, args) != 0)
		return -1;

	if (i2c_set_mode(fd, args) != 0) {
		retval = -1;
		goto end;
	}

	if (i2c_write_reg(fd, args) != 0) {
		retval = -1;
		goto end;
	}

	retval = 0;
end:
	close(fd);
	return retval;
}

int i2c_write_with_struct(struct i2c_rdwr_args args) {
    int retval = 0;
    int fd = -1;

    printf("i2c_num:0x%x, dev_addr:0x%x; reg_addr:0x%x; data:0x%x;"
           "reg_width: %u; data_width: %u.\n",
           args.i2c_num, args.dev_addr , args.reg_addr, args.w_r_union.data,
           args.reg_width, args.data_width);

    if (i2c_open(&fd, args) != 0)
        return -1;

    if (i2c_set_mode(fd, args) != 0) {
        retval = -1;
        goto end;
    }

    if (i2c_write_reg(fd, args) != 0) {
        retval = -1;
        goto end;
    }

    retval = 0;
end:
    close(fd);
    return retval;
}



int gpio_test_out(unsigned int gpio_chip_num, unsigned int gpio_offset_num,unsigned int gpio_out_val)
{
	FILE *fp;
	char file_name[50];
	unsigned char buf[10];
	unsigned int gpio_num;
	gpio_num = gpio_chip_num * 8 + gpio_offset_num;
	sprintf(file_name, "/sys/class/gpio/export");
	fp = fopen(file_name, "w");
	if (fp == NULL) {
		printf("Cannot open %s.\n", file_name);
		return -1;
	}
	fprintf(fp, "%d", gpio_num);
	fclose(fp);
	sprintf(file_name, "/sys/class/gpio/gpio%d/direction", gpio_num);
	fp = fopen(file_name, "rb+");
	if (fp == NULL) {
		printf("Cannot open %s.\n", file_name);
		return -1;
	}
 	fprintf(fp, "out");
	fclose(fp);
	sprintf(file_name, "/sys/class/gpio/gpio%d/value", gpio_num);
	fp = fopen(file_name, "rb+");
	if (fp == NULL) {
		printf("Cannot open %s.\n", file_name);
		return -1;
	}
	if (gpio_out_val)
	strcpy(buf,"1");
	else
	strcpy(buf,"0");
	fwrite(buf, sizeof(char), sizeof(buf) - 1, fp);
	printf("%s: gpio%d_%d = %s\n", __func__,
	gpio_chip_num, gpio_offset_num, buf);
	fclose(fp);
	sprintf(file_name, "/sys/class/gpio/unexport");
	fp = fopen(file_name, "w");
	if (fp == NULL) {
		printf("Cannot open %s.\n", file_name);
		return -1;
	}
	fprintf(fp, "%d", gpio_num);
	fclose(fp);
	return 0;
}

void i2c_pin_set(unsigned int dev_addr){

    // 存储寄存器地址的数组
    unsigned int reg_addresses[] = {
        0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x2A03, 0x3B01, 0x3D02
    };

    // 存储寄存器值的数组
    unsigned int reg_values[] = {
        0x00, 0x08, 0x00, 0x03, 0x00, 0x00, 0x00,
        0x70, 0x00, 0x10, 0x00, 0x07, 0x00, 0x00, 0x00,
        0x70, 0x00, 0x10, 0x03, 0x0f, 0x00, 0x00, 0x00,
        0x70, 0x00, 0x14, 0x03, 0x02, 0x00, 0x00, 0x00,
        0x70, 0x00, 0x00, 0x04, 0x01, 0x00, 0x00, 0x00,
        0x70, 0x00, 0x04, 0x04, 0x00, 0x00, 0x00, 0x00,
        0x70, 0x00, 0x08, 0x04, 0x11, 0x00, 0x00, 0x00,
        0x70, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x70, 0x00, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00,
        0x70, 0x00, 0x14, 0x0C, 0x50
    };

    // 数组元素个数
    size_t num_elements = sizeof(reg_addresses) / sizeof(reg_addresses[0]);

    // 假设的 i2c_num、dev_addr、reg_width、data_width 和 reg_step
    unsigned int i2c_num = 7;
    //unsigned int dev_addr = 0xAA;
    unsigned int reg_width = 2;
    unsigned int data_width = 1;
    unsigned int reg_step = 1;
	dev_addr >>= 1;

    for (size_t i = 0; i < num_elements; i++) {
        struct i2c_rdwr_args args = {
            .i2c_num = i2c_num,
            .dev_addr = dev_addr,
            .reg_addr = reg_addresses[i],
            .w_r_union.data = reg_values[i],
            .reg_width = reg_width,
            .data_width = data_width,
            .reg_step = reg_step
        };

        int result = i2c_write_with_struct(args);
        if (result == 0) {
            printf("I2C write to address 0x%x with value 0x%x succeeded.\n", reg_addresses[i], reg_values[i]);
        } else {
            printf("I2C write to address 0x%x with value 0x%x failed.\n", reg_addresses[i], reg_values[i]);
        }
    }

}

void i2c_light_screen_flip(unsigned int dev_addr){
    // 存储寄存器地址的数组
    unsigned int reg_addresses[] = {
        0x2A03, 0x3B01, 0x0001
    };

    // 存储寄存器值的数组
    unsigned int reg_values[] = {
        0x14, 0x0C, 0x33 
    };

	// 0x2A03, 0x3B01, 0x0001 //color bar screen

	//0x14, 0x0C, 0x33 //inner screen
	//0x14, 0x0C, 0x35 //color bar screen

    // 数组元素个数
    size_t num_elements = sizeof(reg_addresses) / sizeof(reg_addresses[0]);

    // 假设的 i2c_num、dev_addr、reg_width、data_width 和 reg_step
    unsigned int i2c_num = 7;
    //unsigned int dev_addr = 0xAA;
    unsigned int reg_width = 2;
    unsigned int data_width = 1;
    unsigned int reg_step = 1;
	dev_addr >>= 1;

    for (size_t i = 0; i < num_elements; i++) {
        struct i2c_rdwr_args args = {
            .i2c_num = i2c_num,
            .dev_addr = dev_addr,
            .reg_addr = reg_addresses[i],
            .w_r_union.data = reg_values[i],
            .reg_width = reg_width,
            .data_width = data_width,
            .reg_step = reg_step
        };

        int result = i2c_write_with_struct(args);
        if (result == 0) {
            printf("I2C write to address 0x%x with value 0x%x succeeded.\n", reg_addresses[i], reg_values[i]);
        } else {
            printf("I2C write to address 0x%x with value 0x%x failed.\n", reg_addresses[i], reg_values[i]);
        }
    }
}

void i2c_light_screen(unsigned int dev_addr){
    // 存储寄存器地址的数组
    unsigned int reg_addresses[] = {
        0x2A03, 0x3B01, 0x0001 ,0x0101
    };

    // 存储寄存器值的数组
    unsigned int reg_values[] = {
        0x14, 0x0C, 0x33 ,0x03
    };

	// 0x2A03, 0x3B01, 0x0001 //color bar screen

	//0x14, 0x0C, 0x33 //inner screen
	//0x14, 0x0C, 0x35 //color bar screen

    // 数组元素个数
    size_t num_elements = sizeof(reg_addresses) / sizeof(reg_addresses[0]);

    // 假设的 i2c_num、dev_addr、reg_width、data_width 和 reg_step
    unsigned int i2c_num = 7;
    //unsigned int dev_addr = 0xAA;
    unsigned int reg_width = 2;
    unsigned int data_width = 1;
    unsigned int reg_step = 1;
	dev_addr >>= 1;

    for (size_t i = 0; i < num_elements; i++) {
        struct i2c_rdwr_args args = {
            .i2c_num = i2c_num,
            .dev_addr = dev_addr,
            .reg_addr = reg_addresses[i],
            .w_r_union.data = reg_values[i],
            .reg_width = reg_width,
            .data_width = data_width,
            .reg_step = reg_step
        };

        int result = i2c_write_with_struct(args);
        if (result == 0) {
            printf("I2C write to address 0x%x with value 0x%x succeeded.\n", reg_addresses[i], reg_values[i]);
        } else {
            printf("I2C write to address 0x%x with value 0x%x failed.\n", reg_addresses[i], reg_values[i]);
        }
    }
}

//print
void i2c_read_print(unsigned int dev_addr)
{

	int fd = -1;

    // 存储寄存器地址的数组
    unsigned int reg_addresses[] = {
        0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x4200, 0x4201, 0x4100, 0x4101, 0x4102, 0x4103,
        0x6100, 0x6100, 0x2A03, 0x3B01, 0x3D02
    };

    // 数组元素个数
    size_t num_elements = sizeof(reg_addresses) / sizeof(reg_addresses[0]);

    // 假设的 i2c_num、dev_addr、reg_width、data_width 和 reg_step
    unsigned int i2c_num = 7;
    //unsigned int dev_addr = 0xAB;
    unsigned int reg_width = 2;
    unsigned int data_width = 1;
    unsigned int reg_step = 1;
	dev_addr >>= 1;

    for (size_t i = 0; i < num_elements; i++) {
		unsigned int address_end = reg_addresses[i];
        struct i2c_rdwr_args args = {
            .i2c_num = i2c_num,
            .dev_addr = dev_addr,
            .reg_addr = reg_addresses[i],
            .w_r_union.data = 0,
			.w_r_union.reg_addr_end = address_end,
            .reg_width = reg_width,
            .data_width = data_width,
            .reg_step = reg_step
        };

        if (i2c_open(&fd, args) != 0){
            printf("0x%x open failure.\n",reg_addresses[i]);
            continue;
        }
            

        if (i2c_set_mode(fd, args) != 0) {
            printf("0x%x set mode failure.\n",reg_addresses[i]);
            goto end;
        }

        if (i2c_read_regs(fd, args) != 0) {
            printf("0x%x read register failure.\n",reg_addresses[i]);
            goto end;
        }

    end:
	    close(fd);

        printf("read print i2c_num:0x%x, dev_addr:0x%x; reg_addr:0x%x; reg_addr_end:0x%x;"
            "reg_width: %u; data_width: %u; reg_step: %u. \n\n",
            args.i2c_num, args.dev_addr , args.reg_addr, args.w_r_union.reg_addr_end,
            args.reg_width, args.data_width, args.reg_step);
    }

	return 0;
}

td_s32 main(td_s32 argc, td_char *argv[])
{

    td_s32 ret = 0;
    
    printf("Guozhao.37inch Screen start.\n");
    
	
#if ARM_BOARD_TYPE == Hisi_Offical
	//pull down
    gpio_test_out(6,7,0);//RESET1 GPIO6_7
    gpio_test_out(6,5,0);//RESET2 GPIO6_5
    gpio_test_out(7,1,0);//VEE GPIO7_1
    gpio_test_out(7,2,0);//VAN GPIO7_2
    printf("pull down voltage done.\n");

    //5ms
    usleep(5000); /* sleep 5000 us */
    //VAN GPIO7_2
    gpio_test_out(7,2,1);
    //5ms
    usleep(5000); /* sleep 5000 us */
	
    //RESET1 GPIO6_7
    //RESET2 GPIO6_5
    gpio_test_out(6,7,1);
    gpio_test_out(6,5,1);
    //5ms
    usleep(5000); /* sleep 5000 us */
    //VEE GPIO7_1
    gpio_test_out(7,1,1);
    //5ms
    usleep(5000); /* sleep 5000 us */
    printf("pull up voltage done.\n");
#endif
#if ARM_BOARD_TYPE == Self_Devp
	//pull down
    gpio_test_out(2,5,0);//POWER
    gpio_test_out(6,7,0);//RESET1 RESET2
    printf("pull down voltage done.\n");
    //5ms
    usleep(5000); /* sleep 5000 us */

	gpio_test_out(2,5,1);//POWER
	usleep(5000);
	gpio_test_out(6,7,1);
	usleep(5000);
	gpio_test_out(6,7,0);
	usleep(5000);
	gpio_test_out(6,7,1);
	usleep(5000);

#endif
    

    //TODO I2C
    //5ms
    usleep(5000); /* sleep 5000 us */    
	i2c_pin_set(0xA8);

    //TODO MIPI
    //5ms
    usleep(5000); /* sleep 5000 us */

    i2c_light_screen(0xA8);

	//TODO I2C
	usleep(5000); /* sleep 5000 us */    
	i2c_pin_set(0xAA);

	//TODO MIPI
    usleep(5000); /* sleep 5000 us */

    i2c_light_screen_flip(0xAA);
	usleep(5000);

    printf("Guozhao.37inch light done.\n");

	i2c_read_print(0xA9);
	i2c_read_print(0xAB);
    printf("Guozhao.37inch rigester read done.\n");

    //TODO release
    printf("Guozhao.37inch Screen end.\n");

    return ret;
}



// #ifdef __cplusplus
// #if __cplusplus
// }
// #endif
// #endif /* __cplusplus */