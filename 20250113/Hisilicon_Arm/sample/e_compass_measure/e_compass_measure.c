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
        printf("\n");
        usleep(500000);

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
                printf("acc:%.3f %.3f %.3f\r\n", fAcc[0], fAcc[1], fAcc[2]);
                s_cDataUpdate &= ~ACC_UPDATE;
            }
            if(s_cDataUpdate & GYRO_UPDATE)
            {
                printf("gyro:%.3f %.3f %.3f\r\n", fGyro[0], fGyro[1], fGyro[2]);
                s_cDataUpdate &= ~GYRO_UPDATE;
            }
            if(s_cDataUpdate & ANGLE_UPDATE)
            {
                printf("angle:%.3f %.3f %.3f\r\n", fAngle[0], fAngle[1], fAngle[2]);
                s_cDataUpdate &= ~ANGLE_UPDATE;
            }
            if(s_cDataUpdate & MAG_UPDATE)
            {
                printf("mag:%d %d %d\r\n", sReg[HX], sReg[HY], sReg[HZ]);
                s_cDataUpdate &= ~MAG_UPDATE;
            }
        }
    }
    
    printf("e compass UART read thread exiting\n");
    return NULL;
}

int main() {

    //打开电子罗盘测距串口
    char file_name[I2C_NAME_SIZE];
	if (sprintf_s(file_name, I2C_NAME_SIZE, "/dev/ttyAMA2") < 0) {
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
    
    while (e_compass_running) {
        usleep(10000);
    }

    // 清理资源
    e_compass_running = 0;
    pthread_join(e_compass_uart_thread, NULL);
    close(e_compass_fd);
    printf("e compass program exited successfully\n");
    return 0;
}
