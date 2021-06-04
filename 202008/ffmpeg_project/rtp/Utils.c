#include "Utils.h"
#include <stdio.h>

int readFile(uint8_t **stream ,int *len ,const char *file) {
	FILE* fp = NULL;
	long size = 0;
	uint8_t* buf;
	printf("Read File %s\n" ,file);
	fp = fopen(file ,"r");
	if (!fp) {
		return -1;
	}

	fclose(fp);
	prinf();
}