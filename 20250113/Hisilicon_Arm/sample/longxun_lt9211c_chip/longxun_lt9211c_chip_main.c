//Hisi_Offical Self_Devp
#define Hisi_Offical 0
#define Self_Devp 1
#define ARM_BOARD_TYPE Self_Devp

#include "include.h"

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
		printf("\nsprintf_s error!\n");
		return -1;
	}

	*fd = open(file_name, O_RDWR);
	if (*fd < 0) {
		printf("\nOpen %s error!\n", file_name);
		return -1;
	}
	return 0;
}

static int i2c_set_mode(int fd,  struct i2c_rdwr_args args)
{
	if (fd < 0)
		return -1;
	if (ioctl(fd, I2C_SLAVE_FORCE, args.dev_addr)) {
		printf("\nset i2c device address error!\n");
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
		printf("\nmemset_s fail!\n");
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

static int i2c_read_regs(int fd, struct i2c_rdwr_args args ,unsigned char* p_data)
{
	unsigned int cur_addr;
	unsigned int data = -1;
	unsigned char buf[I2C_READ_BUF_LEN];
	static struct i2c_rdwr_ioctl_data rdwr;
	static struct i2c_msg msg[I2C_MSG_CNT];

	rdwr.msgs = &msg[0];
	rdwr.nmsgs = (__u32)I2C_MSG_CNT;
	if (i2c_ioc_init(&rdwr, buf, sizeof(buf), args) != 0){
		printf("\nioc error.\n");
		return -1;
	}

	//printf("\nioc start.\n");
	//printf("\nioc addr 04x%x: 04x%x\n", args.reg_addr, args.w_r_union.reg_addr_end);
	for (cur_addr = args.reg_addr; cur_addr <= args.w_r_union.reg_addr_end; cur_addr += args.reg_step) {
		if (args.reg_width == BYTE_CNT_2) {
			buf[0] = (cur_addr >> BIT_CNT_8) & 0xff;
			buf[1] = cur_addr & 0xff;
		} else {
			buf[0] = cur_addr & 0xff;
		}

		//printf("\nioc shikeDebug.\n");

		if (ioctl(fd, I2C_RDWR, &rdwr) != I2C_READ_STATUS_OK) {
			//printf("\nCMD_I2C_READ error!\n");
			return -1;
		}

		if (args.data_width == BYTE_CNT_2)
			data = buf[1] | (buf[0] << BIT_CNT_8);
		else
			data = buf[0];	
		//printf("\n0x%x: 0x%x\n", cur_addr, data);
	} 
	*p_data = data & 0xff;
	//printf("\nioc end.\n");
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
		//printf("\ni2c write error!\n");
		return -1;
	}

	return 0;
}

int i2c_read(int argc, char *argv[])
{
	int retval = 0;
	int fd = -1;

	unsigned int i2c_num_type = 1;
#if ARM_BOARD_TYPE == Hisi_Offical
	i2c_num_type = 1;
#endif
#if ARM_BOARD_TYPE == Self_Devp
	i2c_num_type = 7;
#endif

	struct i2c_rdwr_args args = {
		.i2c_num = i2c_num_type,
		.dev_addr = 0,
		.reg_addr = 0,
		.w_r_union.reg_addr_end = 0,
		.reg_width = 1,
		.data_width = 1,
		.reg_step = 1
	};

	// if (parse_args(argc, argv, &args) != 0) {
	// 	print_r_usage();
	// 	return -1;
	// }

	// printf("\ni2c_num:0x%x, dev_addr:0x%x; reg_addr:0x%x; reg_addr_end:0x%x;"
	// 	"reg_width: %u; data_width: %u; reg_step: %u. \n",
	// 	args.i2c_num, args.dev_addr , args.reg_addr, args.w_r_union.reg_addr_end,
	// 	args.reg_width, args.data_width, args.reg_step);

	if (i2c_open(&fd, args) != 0)
		return -1;

	if (i2c_set_mode(fd, args) != 0) {
		retval = -1;
		goto end;
	}
	unsigned char data = 0;
	if (i2c_read_regs(fd, args ,&data) != 0) {
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

	unsigned int i2c_num_type = 1;
#if ARM_BOARD_TYPE == Hisi_Offical
	i2c_num_type = 1;
#endif
#if ARM_BOARD_TYPE == Self_Devp
	i2c_num_type = 7;
#endif

struct i2c_rdwr_args args = {
		.i2c_num = i2c_num_type,
		.dev_addr = 0,
		.reg_addr = 0,
		.w_r_union.data = 0,
		.reg_width = 1,
		.data_width = 1,
	};

	// if (parse_args(argc, argv, &args) != 0) {
	// 	print_w_usage();
	// 	return -1;
	// }

	// printf("\ni2c_num:0x%x, dev_addr:0x%x; reg_addr:0x%x; data:0x%x;"
	// 	"reg_width: %u; data_width: %u.\n",
	// 	args.i2c_num, args.dev_addr , args.reg_addr, args.w_r_union.data,
	// 	args.reg_width, args.data_width);

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

    // printf("\ni2c_num:0x%x, dev_addr:0x%x; reg_addr:0x%x; data:0x%x;"
    //        "reg_width: %u; data_width: %u.\n",
    //        args.i2c_num, args.dev_addr , args.reg_addr, args.w_r_union.data,
    //        args.reg_width, args.data_width);

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

int gpio_test_out_copy(unsigned int gpio_chip_num, unsigned int gpio_offset_num,unsigned int gpio_out_val)
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

u16  ucI2cAddr = 0x5a;

u8 HDMI_ReadI2C_Byte(u16 RegAddr)
{
	usleep(10);
    u8 p_data = 0;
    int fd = -1;

	unsigned int i2c_num_type = 1;
#if ARM_BOARD_TYPE == Hisi_Offical
	i2c_num_type = 1;
#endif
#if ARM_BOARD_TYPE == Self_Devp
	i2c_num_type = 7;
#endif

    // 构造I2C读写参数结构体，注意reg_width=2（2字节寄存器地址）
    struct i2c_rdwr_args args = {
        .i2c_num = i2c_num_type,                // I2C总线编号
        .dev_addr = ucI2cAddr >> 1,  // 设备地址右移1位（符合I2C协议）
        .reg_addr = RegAddr,    // 将u8类型的RegAddr扩展为16位
		.w_r_union.reg_addr_end = RegAddr,
        //.w_r_union.data = 0,         // 读操作时数据字段无效
        .reg_width = 1,              // 寄存器地址宽度为2字节
        .data_width = 1,             // 读取数据宽度为1字节
        .reg_step = 1                // 寄存器地址自增步长
    };

	//printf("\nI2CAddr: %04X Register=0x%04X\n",(u16)(ucI2cAddr >> 1), (u16)RegAddr);

    if (i2c_open(&fd, args) != 0) {
        //printf("\nI2C open failed for RegAddr 0x%02X\n", RegAddr);
        return 0;
    }

    if (i2c_set_mode(fd, args) != 0) {
        //printf("\nI2C set mode failed for RegAddr 0x%02X\n", RegAddr);
        close(fd);
        return 0;
    }

    // 执行读操作
    if (i2c_read_regs(fd, args ,&p_data) != 0) {
        //printf("\nI2C read failed for RegAddr 0x%02X\n", RegAddr);
        close(fd);
        return 0;
    }

    close(fd);
    //printf("\n  p_data=0x%02X\n", p_data);
	//printf("\nI2C Read: Device=0x%02X, Register=0x%04X, Data=0x%02X\n", 
    //       ucI2cAddr, args.reg_addr, p_data);
    return p_data;
}

bool HDMI_WriteI2C_Byte(u16 RegAddr, u16 d)
{
	usleep(10);
    bool flag = false;
    
	unsigned int i2c_num_type = 1;
#if ARM_BOARD_TYPE == Hisi_Offical
	i2c_num_type = 1;
#endif
#if ARM_BOARD_TYPE == Self_Devp
	i2c_num_type = 7;
#endif

    // 构造I2C读写参数结构体，注意reg_width=2（2字节寄存器地址）
    struct i2c_rdwr_args args = {
        .i2c_num = i2c_num_type,                // I2C总线编号
        .dev_addr = ucI2cAddr >> 1,  // 设备地址右移1位
        .reg_addr = RegAddr,    // 将u8类型的RegAddr扩展为16位
        .w_r_union.data = d,         // 要写入的数据
        .reg_width = 1,              // 寄存器地址宽度为2字节
        .data_width = 1,             // 写入数据宽度为1字节
        .reg_step = 1                // 寄存器地址自增步长
    };

    // 执行写操作
    int result = i2c_write_with_struct(args);
    if (result == 0) {
        // printf("\nI2C Write: Device=0x%02X, Register=0x%04X, Data=0x%02X succeeded\n", 
        //         ucI2cAddr, args.reg_addr, d);
        flag = true;
    } else {
        // printf("\nI2C Write : Device=0x%02X, Register=0x%04X, Data=0x%02X failed\n", 
        //         ucI2cAddr, args.reg_addr, d);
        flag = false;
    }
    
    return flag;
}

td_s32 main(td_s32 argc, td_char *argv[])
{

    td_s32 ret = 0;
    
    printf("\nLongxun lt9211c start.\n");
    
#if ARM_BOARD_TYPE == Self_Devp
	//MIPI电子切换开关切换到LT9211C  GPIO6_5拉低
	gpio_test_out_copy(6,5,0);
	usleep(2000000);
#endif

	Mod_System_Init();
    Mod_LT9211C_Reset();
    ////LTLog(LOG_INFO, "LT9211C %s %s",__DATE__,__TIME__);
    Mod_ChipID_Read();
    
#if (LT9211C_MODE_SEL != PATTERN_OUT)
    Mod_SystemRx_PowerOnInit();
    Mod_SystemTx_PowerOnInit();
#endif

    while(1)
    {
        #if (LT9211C_MODE_SEL == PATTERN_OUT)
                Mod_ChipTx_PtnOut();
        #endif

		#if (LT9211C_MODE_SEL == MIPI_REPEATER)
        Mod_MipiRpt_Handler();
        #endif

        #if (LT9211C_MODE_SEL == MIPI_IN_MIPI_OUT)
        Mod_MipiRx_Handler();
        Mod_MipiTx_Handler();
        #endif

    }

    //TODO release
    printf("\nLongxun lt9211c end.\n");

    return ret;
}



// #ifdef __cplusplus
// #if __cplusplus
// }
// #endif
// #endif /* __cplusplus */