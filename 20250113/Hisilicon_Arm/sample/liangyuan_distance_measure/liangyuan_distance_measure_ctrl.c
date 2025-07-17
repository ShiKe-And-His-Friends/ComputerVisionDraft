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

// 共享资源结构体
typedef struct {
    pthread_mutex_t mutex;      // 互斥锁
    pthread_cond_t cond;        // 条件变量
    int button_id;              // 按钮编号
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

    // 初始化事件缓冲区
    pthread_mutex_init(&event_buffer.lock, NULL);
    pthread_cond_init(&event_buffer.cond, NULL);
    
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

//============================================================================
//              以上是button
//============================================================================

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

void print_binary(unsigned char byte) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (byte >> i) & 1);
    }
    printf("  ");
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
        }
        
        // 短暂延时，避免CPU占用过高
        usleep(10000); // 10ms
    }
    
    printf("liangyuan measure distance UART read thread exiting\n");
    return NULL;
}

int main() {

    // 初始化互斥锁和条件变量
    pthread_mutex_init(&g_event.mutex, NULL);
    pthread_cond_init(&g_event.cond, NULL);
    g_event.button_id = 0;
    g_event.exit_flag = 0;

    // 创建子线程
    pthread_t monitor_thread;
    if (pthread_create(&monitor_thread, NULL, button_monitor_thread, NULL) != 0) {
        perror("Failed to create button monitor thread");
    }


    //打开亮源激光测距串口
    char file_name[I2C_NAME_SIZE];
	if (sprintf_s(file_name, I2C_NAME_SIZE, "/dev/ttyAMA2") < 0) {
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
    
    // // 主线程循环
    // while(ly_measure_running) {
    //     //接收按键后
    //     //写串口
    //     char write_buffer[32];
    //     sprintf(write_buffer, "55 01 02 00 00 56\n");

    //     // 加锁保护串口写入
    //     pthread_mutex_lock(&ly_measure_uart_mutex);
    //     int bytes_written = write(ly_measure_fd, write_buffer, strlen(write_buffer));
    //     pthread_mutex_unlock(&ly_measure_uart_mutex);

    //     if (bytes_written < 0) {
    //         perror("liangyuan measure distance UART write error");
    //     } else {
    //         printf("liangyuan measure distance sent %d bytes to UART\n", bytes_written);
    //     }
    // }

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

        // 执行按钮对应的操作
        switch (button_state)
        {
            //红外视频/可见光视频
            case 2:
                // // 根据当前状态执行相应函数
                // if (g_event.state == STATE_A) {
                //     // 释放互斥锁，允许按钮线程在执行函数A时更新状态
                //     g_event.state = STATE_B;
                //     Close_Infrared_Vio();
                //     Open_Colorcamera_Vio();
                // } else if (g_event.state == STATE_B) {
                //     g_event.state = STATE_A;
                //     Close_Colorcamera_Vio();
                //     Open_Infrared_Vio();
                // } else {
                // }
                break;

            //变倍&变焦+  按键按下
            case 5:
                printf("+++ 111 ...\n");
                // set_zoom_state(ZOOM_INCREASE);
                // printf("Colorcamera Zooming in...\n");
                break;

            //变倍&变焦+  按键抬起
            case 6:
                printf("+++ 222...\n");
                // set_zoom_state(ZOOM_STOP);
                // printf("Colorcamera Zoom stopped at %.1fx\n", get_current_magnification());
                break;

            //变倍&变焦-  按键按下
            case 9:
                printf("--- 111...\n");
                // set_zoom_state(ZOOM_DECREASE);
                // printf("Colorcamera Zooming out...\n");
                        //接收按键后
                //写串口
                char write_buffer[32];
                sprintf(write_buffer, "55 01 02 00 00 56\n");

                // 复制参数字符串用于处理
                unsigned  char hex_str[256];
                strncpy(hex_str, write_buffer, sizeof(hex_str) - 1);
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

            //变倍&变焦-  按键抬起
            case 10:
                printf("--- 222...\n");
                // set_zoom_state(ZOOM_STOP);
                // printf("Colorcamera Zoom stopped at %.1fx\n", get_current_magnification());
                break;
            
            default:
                break;
        }
    }








    
    // 清理资源
    ly_measure_running = 0;
    pthread_join(ly_measure_uart_thread, NULL);
    close(ly_measure_fd);
    pthread_mutex_destroy(&ly_measure_uart_mutex);
    printf("liangyuan measure distance program exited successfully\n");

    return 0;
}
