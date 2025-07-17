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

/**
 * @brief 以hexdump格式打印二进制数据（十六进制+ASCII）
 * @param data 数据缓冲区
 * @param len 数据长度（字节数）
 */
 void hexdump(const unsigned char *data, size_t len) {
    size_t i, j;
    const size_t bytes_per_line = 16;  // 每行显示16字节（可改为8/16/32）

    for (i = 0; i < len; i += bytes_per_line) {
        // 打印行起始地址（可选，此处省略，可添加i的十六进制地址）
        // printf("%08zx  ", i);

        // 打印十六进制部分
        for (j = 0; j < bytes_per_line; j++) {
            if (i + j < len) {
                printf("%02X ", data[i + j]);  // 两位十六进制
            } else {
                printf("   ");  // 不足一行时用空格填充
            }
            if (j == bytes_per_line / 2 - 1) {
                printf(" ");  // 中间添加空格分隔（如8字节一组）
            }
        }

        // 打印ASCII部分分隔符
        printf(" | ");

        // 打印ASCII字符（可打印字符显示，否则显示'.'）
        for (j = 0; j < bytes_per_line; j++) {
            if (i + j < len) {
                unsigned char c = data[i + j];
                printf("%c", (isprint(c) ? c : '.'));  // 可打印字符直接显示
            }
        }

        printf("\n");
    }
}

int uart_set(int fd)
{
    struct termios options;

    if(tcgetattr(fd,&options) < 0) {
        printf("tcgetattr error\n");
        return -1;
    }
    //设置波特率
    cfsetispeed(&options,B9600);
    cfsetospeed(&options,B9600);

    //cfsetispeed(&options,B115200);
    //cfsetospeed(&options,B115200);

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
    
    if(tcsetattr(fd,TCSANOW,&options) < 0) {
        printf("tcsetattr failed\n");
        return -1;
    }
    return 0;
}

void print_binary(unsigned char byte) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (byte >> i) & 1);
    }
    printf("  ");
}

int uart_read(int fd,unsigned char *buf, int len)
{
    printf("uart start.\n");

    int ret;
    int read_num, left_num;
    fd_set rfds;
    char *ptr;
    struct timeval timeout;

    FD_ZERO(&rfds);
    FD_SET(fd,&rfds);

    // 设置5秒超时
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    left_num = len;

    printf("select start.\n");
    ret = select(fd+1,&rfds,NULL,NULL, &timeout);
    printf("select end.\n");
    if (ret > 0) {
        while(left_num > 0)
        {
            printf("read start.\n");
            read_num = read(fd,buf,left_num);
            printf("read end %d\n",read_num);

            if (read_num > 0) {
                left_num -= read_num;
                ptr += read_num;
            } else if (read_num == 0) {
                printf("connection closed\n");
                break;
            } else {
                perror("connection failed");
                return -1;
            }
        }
    } else if (ret < 0) {
        perror("select error");
        return -1;
    } else if (ret == 0) {
        printf("select timeout\n");
        return 0;  // 超时返回0
    }

    printf("uart end.\n");
    return len - left_num;
}

int uart_write(int fd,unsigned char *buf, int len)
{
    int ret;
    int write_num, left_num;
    unsigned char *ptr;

    left_num = len;
    while(left_num > 0)
    {
        write_num = write(fd,(unsigned char *)buf,left_num);
        if (write_num > 0) {
            left_num -= write_num;
            ptr += write_num;
            printf("write num %d\n" ,write_num);
        } else {
            printf("write fail!\n");
            return -1;
        }
    }
    return 0;
}

// 移除字符串中的空格
void remove_spaces(unsigned char *str)
{
    unsigned  char *src = str;
    unsigned  char *dst = str;
    
    while (*src) {
        if (!isspace((unsigned char)*src)) {
            *dst = *src;
            dst++;
        }
        src++;
    }
    *dst = '\0';  // 添加字符串结束符
}

// 将十六进制字符串转换为二进制数据
int hex_string_to_bytes(const unsigned char *hex_str, unsigned char *bytes, int max_len)
{
    int len = strlen(hex_str);
    int byte_count = 0;
    unsigned char temp;
    
    // 检查十六进制字符串长度是否为偶数
    if (len % 2 != 0) {
        fprintf(stderr, "Hex string length must be even\n");
        return -1;
    }
    
    // 检查是否超出最大长度
    if (len / 2 > max_len) {
        fprintf(stderr, "Hex string too long\n");
        return -1;
    }
    
    // 转换每对十六进制字符为一个字节
    for (int i = 0; i < len; i += 2) {
        if (sscanf(&hex_str[i], "%2hhx", &temp) != 1) {
            fprintf(stderr, "Invalid hex character at position %d\n", i);
            return -1;
        }
        bytes[byte_count++] = temp;
    }
    
    return byte_count;
}

#define HEX_HEAD_LEN 2
#define DECIMAL 10
#define HEXADECIMAL 16
int str2number(const char *str, unsigned long *number)
{
	const char *start_ptr = NULL;
	char *end_ptr = NULL;
	int radix;

	if (str == NULL || number == NULL)
		return -1;

	if (*str == '0' && (*(str + 1) == 'x' || *(str + 1) == 'X')) {
		if (*(str + HEX_HEAD_LEN) == '\0')
			return -1;
		start_ptr = str + HEX_HEAD_LEN;
		radix = HEXADECIMAL;
	} else {
		start_ptr = str;
		radix = DECIMAL;
	}

	*number = strtoul(start_ptr, &end_ptr, radix);
	if ((start_ptr + strlen(start_ptr)) != end_ptr)
		return -1;

	return 0;
}

td_s32 main(td_s32 argc, td_char *argv[])
{
	if (argc != 3) {
		printf("Lingdong Camera usage error.\n");
		return -1;
	}
	printf("Lingdong Camera start.\n");
	
	unsigned long val;
	if (str2number(argv[1], &val) != 0)
	{
		printf("Lingdong Camera number format error.\n");
		return -1;
	}
	printf("Lingdong Camera number format done.\n");

	char file_name[I2C_NAME_SIZE];
	if (sprintf_s(file_name, I2C_NAME_SIZE, "/dev/ttyAMA%u", val) < 0) {
		printf("sprintf_s error!\n");
		return -1;
	}

	printf("%s\n",file_name);
	printf("sprintf_s done!\n");

	int fd = open(file_name, O_RDWR|O_NOCTTY);//O_RDWR|O_NOCTTY|O_NDELAY
	if (fd < 0) {
		printf("Open %s error!\n", file_name);
		return -1;
	}
    
	printf("Open %s success.\n", file_name);
    
	// 配置UART
    if (uart_set(fd) < 0) {
        close(fd);
		printf("set UART error!\n");
        return -1;
    }
	printf("set UART success.\n");
    
    // 复制参数字符串用于处理
    unsigned  char hex_str[256];
    strncpy(hex_str, argv[2], sizeof(hex_str) - 1);
    hex_str[sizeof(hex_str) - 1] = '\0';
    
    // 移除空格
    remove_spaces(hex_str);
    printf("Processed hex string: %s\n", hex_str);
    
    // 转换为二进制数据
    unsigned char bin_data[128];
    int bin_len = hex_string_to_bytes(hex_str, bin_data, sizeof(bin_data));
    if (bin_len < 0) {
        close(fd);
        return -1;
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

    // 发送数据（使用strlen而非sizeof）
    if (uart_write(fd, (unsigned char *)bin_data, bin_len) < 0) {
        close(fd);
		printf("send UART error!\n");
        return -1;
    }
    
	int m = 5;
	while(m-- >0){
		// 读取响应（如果有）
		unsigned char response[256];
		int bytes_read = uart_read(fd, response, sizeof(response) - 1);
		if (bytes_read > 0) {
			printf("Received %d bytes:\n", bytes_read);
            hexdump((const unsigned char*)response, (size_t)bytes_read);
            
            //response[bytes_read] = '\0';  // 添加字符串结束符
			//printf("Received %d bytes: %s\n", bytes_read, response);
            break;
		} else if (bytes_read == 0) {
			printf("No data received(%d)\n",m);
		} else {
			printf("Error reading data\n");
		}
		usleep(1);
	}

    //5ms
    //usleep(5000); /* sleep 5000 us */

	close(fd);
    printf("Lingdong Camera end.\n");

    return 0;
}
