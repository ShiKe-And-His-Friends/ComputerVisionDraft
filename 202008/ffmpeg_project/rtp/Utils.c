#include "Utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int readFile(uint8_t **stream ,int *len ,const char *file) {
	FILE* fp = NULL;
	long size = 0;
	uint8_t* buf;
	printf("read file %s\n" ,file);
	fp = fopen(file ,"r");
	if (!fp) {
		printf("read file failed.\n");
		return -1;
	}

	fseek(fp ,0L ,SEEK_END);
	size = ftell(fp);
	fseek(fp, 0L, SEEK_SET);

	printf("file size is %d\n" ,size);

	buf = (uint8_t*)malloc(size * sizeof(uint8_t));
	memset(buf ,0 ,size);

	long mysize = fread(buf,size , 1, fp);
	printf("read file size change is %ld\n" ,mysize);
	if (mysize != size) {
		printf("read error.\n");
		return -1;
	}
	printf("closed file.\n");
	fclose(fp);
	return 0;
}



//int readFile(uint8_t** stream, int* len, const char* file) {
//    FILE* fp = NULL;
//    size_t size = 0;
//    uint8_t* buf;
//
//    printf("readFile %s\n", file);
//    fp = fopen(file, "r");
//    if (!fp)
//        return -1;
//
//#if 1
//
//    fseek(fp, 0L, SEEK_SET);
//    size = ftell(fp);
//    printf("file length 1  %d\n", size);
//
//    fseek(fp, 0L, SEEK_END);
//    size = ftell(fp);
//    printf("file length 2  %d\n", size);
//
//    fseek(fp, 0L, SEEK_SET);
//    /*
//    size = ftell(fp);
//    printf("file length 3  %d\n", size);*/
//    //size = 100;
//#else
//    struct stat info = { 0 };
//    stat(file, &info);
//    size = info.st_size;
//#endif
//
//    buf = (uint8_t*)(malloc((size) * sizeof(uint8_t)));
//    memset(buf, 0, (size_t)(size));
//
//    long read_size = 0;
//    read_size = fread(buf, (size_t)(size), 1, fp);
//    //if (read_size != size) {
//    //    printf("read err %d %d\n" , read_size ,size);
//    //    return -1;
//    //}
//
//    int i = 0;
//    for (i = 0; i < 20; i++) {
//        printf("%d \n", *(buf + size - 10000 - i));
//    }
//    printf("read err %d %d\n" , read_size ,size);
//
//    if (ferror(fp)) {
//        printf("check error true.\n");
//    }
//    else {
//        printf("check error false.\n");
//    }
//
//    if (feof(fp)) {
//        printf("file eof true.\n");
//    }
//    else {
//        printf("file eof false.\n");
//    }
//
//    fclose(fp);
//
//    *stream = buf;
//    *len = (int)size;
//
//    printf("File Size = %d Bytes\n", *len);
//    return 0;
//}
