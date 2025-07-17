#include "IIC_GPIO.h"

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