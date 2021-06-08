#include "Utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t* Load8(uint8_t* p, uint8_t x) {
	*p = x;
	return p + 1;
}

uint8_t* Load16(uint8_t* p, uint16_t x) {
	p = Load8(p, (uint8_t)(x >> 8));
	p = Load8(p ,(uint8_t)x);
	return p;
}

uint8_t* Load32(uint8_t* p, uint32_t x) {
	p = Load16(p ,(uint16_t)(x >> 16));
	p = Load16(p, (uint16_t)x);
	return p;
}

int readFile(uint8_t **stream ,int *len ,const char *file) {
	FILE* fp = NULL;
	size_t size = 0;
	uint8_t* buf;
	printf("read file %s\n" ,file);
	fp = fopen(file ,"rb");
	if (!fp) {
		printf("read file failed.\n");
		return -1;
	}

	fseek(fp ,0L ,SEEK_END);
	size = ftell(fp);
	fseek(fp, 0L, SEEK_SET);
    //size = 10000;

	printf("file size is %d\n" ,size);

	buf = (uint8_t*)malloc(size * sizeof(uint8_t));
    if (!buf) {
        printf("allocate %p failure.\n", buf);
        return -1;
    }
    memset(buf ,0 ,(size_t)size);

	long mysize = fread(buf, 1 ,size, fp);
	if (mysize != size) {
		printf("read space %d done %d.\n" ,size ,mysize);
        if (feof(fp)) {
            printf("file eof(true) true.\n");
        }
        else {
            printf("file eof(true) false.\n");
            return -1;
        }
        if (ferror(fp)) {
            printf("check error(false) true.\n");
			return -1;
        }
        else {
            printf("check error(false) false.\n");
        }
	}
	else {
		printf("read file size change is %ld\n", mysize);
	}

	/*
	int i = 0;
	for (i = 0; i < 20; i++) {
		printf("%d \n", *(buf + size - 10000 - i));
	}
	*/

	*stream = buf;
	*len = (int)size;

	printf("closed file.\n");
	fclose(fp);
	return 0;
}