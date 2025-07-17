#include "uart_camera_ctrl.h"

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

int uart_set(int fd ,int setiospeed)
{
    struct termios options;

    if(tcgetattr(fd,&options) < 0) {
        printf("tcgetattr error\n");
        return -1;
    }
    //设置波特率
    if (setiospeed == 0) {
        cfsetispeed(&options,B9600);
        cfsetospeed(&options,B9600);
    } else {
        cfsetispeed(&options,B115200);
        cfsetospeed(&options,B115200);
    }
     
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
    
    // 刷新输入输出缓冲区
    tcflush(fd, TCIOFLUSH);

    if(tcsetattr(fd,TCSANOW,&options) < 0) {
        printf("tcsetattr failed\n");
        return -1;
    }

    // 设置写入超时（使用O_NONBLOCK和select）
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);

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
    ret = select(fd+1,&rfds,NULL,NULL, &timeout);
    if (ret > 0) {
        while(left_num > 0)
        {
            read_num = read(fd,buf,left_num);

            if (read_num > 0) {
                left_num -= read_num;
                ptr += read_num;
            } else if (read_num == 0) {
                break;
            } else {
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
    return len - left_num;
}

int uart_write(int fd, unsigned char *buf, int len)
{
    int ret;
    int write_num, left_num;
    fd_set writefds;
    struct timeval timeout;
    
    left_num = len;
    unsigned char *ptr = buf;  // 初始化指针
    
    while (left_num > 0) {
        // 设置select超时（例如5秒）
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        
        // 准备select
        FD_ZERO(&writefds);
        FD_SET(fd, &writefds);
        
        // 等待可写或超时
        ret = select(fd + 1, NULL, &writefds, NULL, &timeout);
        
        if (ret < 0) {
            printf("select error: %d\n", errno);
            return -1;
        } else if (ret == 0) {
            printf("write timeout!\n");
            return -1;
        }
        
        // 现在可以写入
        write_num = write(fd, ptr, left_num);
        
        if (write_num > 0) {
            left_num -= write_num;
            ptr += write_num;  // 更新指针位置
        } else if (write_num < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // 暂时不可写，继续等待
                continue;
            } else {
                printf("write error: %d\n", errno);
                return -1;
            }
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

int send_uart_commond(td_char *argv[] ,int setiospeed)
{
	unsigned long val;
	if (str2number(argv[0], &val) != 0)
	{
		printf("Uart commond number format error.\n");
		return -1;
	}

	char file_name[I2C_NAME_SIZE];
	if (sprintf_s(file_name, I2C_NAME_SIZE, "/dev/ttyAMA%u", val) < 0) {
		printf("sprintf_s error!\n");
		return -1;
	}

    int fd = -1;
	fd = open(file_name, O_RDWR|O_NOCTTY);//O_RDWR|O_NOCTTY|O_NDELAY
	if (fd < 0) {
		printf("Open %s error!\n", file_name);
		return -1;
	}

	// 配置UART
    if (uart_set(fd ,setiospeed) < 0) {
        close(fd);
		printf("set UART error!\n");
        goto error_exit;
    }
    
    // 复制参数字符串用于处理
    unsigned  char hex_str[256];
    strncpy(hex_str, argv[1], sizeof(hex_str) - 1);
    hex_str[sizeof(hex_str) - 1] = '\0';
    
    // 移除空格
    remove_spaces(hex_str);
    //printf("Processed hex string: %s\n", hex_str);
    
    // 转换为二进制数据
    unsigned char bin_data[128];
    int bin_len = hex_string_to_bytes(hex_str, bin_data, sizeof(bin_data));
    if (bin_len < 0) {
        printf("Hex string conversion error!\n");
        goto error_exit;
    }
    
    // 打印转换后的二进制数据
    printf("Binary data  %d  to send: ",bin_len);
    for (int i = 0; i < bin_len; i++) {
        printf("%02X ", bin_data[i]);
    }
    // printf("\nbinary: ");
    // for (int i = 0; i < bin_len; i++) {
    //     print_binary(bin_data[i]);
    // }
    printf("\n");

    // 发送数据（使用strlen而非sizeof）
    if (uart_write(fd, (unsigned char *)bin_data, bin_len) < 0) {
        close(fd);
		printf("send UART error!\n");
        goto error_exit;
    }

    /***********************************
    int ret = -2;
	int m = 5;
	while(m-- >0){
		// 读取响应（如果有）
		unsigned char response[256];
		int bytes_read = uart_read(fd, response, sizeof(response) - 1);
		if (bytes_read > 0) {
			printf("Received %d bytes: ", bytes_read);
            hexdump((const unsigned char*)response, (size_t)bytes_read);
            
            //response[bytes_read] = '\0';  // 添加字符串结束符
			//printf("Received %d bytes: %s\n", bytes_read, response);
            ret = 0;
            break;
		}
		usleep(1);
	}
	close(fd);

    return ret;
    *******************************************/

    close(fd);
    return 0;

error_exit:
    // 确保文件描述符被关闭
    if (fd >= 0) {
        close(fd);
    }
    return -1;
}
