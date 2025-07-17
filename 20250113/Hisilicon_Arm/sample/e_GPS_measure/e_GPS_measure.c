/*
  Copyright (c), 2001-2024, Shenshu Tech. Co., Ltd.
 */
//====================================
// 标准C库头文件
//====================================
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

//====================================
// POSIX标准头文件
//====================================
#include <pthread.h>
#include <signal.h>
#include <termios.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

//====================================
// Linux内核头文件
//====================================
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fb.h>

//====================================
// 自定义头文件
//====================================
#include "ot_osal.h"
#include "sample_comm.h"
#include "securec.h"
#include "ot_type.h"
#include "ot_common.h"
#include "ot_defines.h"
#include "osal_ioctl.h"

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


/*
 * Print navigation data and wait for user's keystroke
 * @navdata:    the navigation data
 */
void display_navdata(struct nav_data *navdata) {
    printf("NAVDATA:\n");
    printf("NAVDATA.FIX_VALID = %d\n", navdata->is_fixed);
    printf("NAVDATA.DATE = %d-%02d-%02d\n", navdata->date.year, navdata->date.month, navdata->date.day);
    printf("NAVDATA.TIME= %02d:%02d:%02d.%03d\n", navdata->time.hour, navdata->time.minute, navdata->time.second, navdata->time.ms);
    printf("NAVDATA.LAT = %.6f\n", navdata->lat);
    printf("NAVDATA.LON = %.6f\n", navdata->lon);
    printf("NAVDATA.ALT = %.2f\n", navdata->alt);
    printf("NAVDATA.HEADING = %.2f\n", navdata->heading);
    printf("NAVDATA.SPEED = %.2f\n", navdata->speed);
    printf("NAVDATA.HDOP = %.1f\n", navdata->hdop);
    printf("NAVDATA.VDOP = %.1f\n", navdata->vdop);
    printf("NAVDATA.PDOP = %.1f\n", navdata->pdop);
    printf("NAVDATA.NUM_SV_FIX = %d\n", navdata->sv_inuse);
    printf("NAVDATA.NUM_SV_VIEW = %d\n", navdata->sv_inview);
    
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

int main() {

    //打开GPS定位测距串口
    char file_name[I2C_NAME_SIZE];
	if (sprintf_s(file_name, I2C_NAME_SIZE, "/dev/ttyAMA2") < 0) {
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
    
    while (e_GPS_running) {
        usleep(10000);
    }

    // 清理资源
    e_GPS_running = 0;
    pthread_join(e_GPS_uart_thread, NULL);
    close(e_GPS_fd);
    printf("e GPSprogram exited successfully\n");
    return 0;
}
