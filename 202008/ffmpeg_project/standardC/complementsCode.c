#include <stdio.h>
#include <string.h>
#define MAX 255

int compoments_codes() {
	char a = -1;
	signed char b = -1;
	unsigned char c = -1;
	printf("a=%d ,b=%d ,c=%d" ,a ,b ,c);
	return 0;
}

int death_loop() {
	unsigned char a[MAX], i;
	for (i = 0; i < MAX ; i++) {
		a[i] = i;
		for (i = 0; i < MAX;i++) {
			printf("%d ",a[i]);
		}
	}
	return 0;
}

int compolemt_codes_add() {
	char a = -128;
	char b = 128;
	printf("a=%u ,b=%u" ,a ,b);
	return 0;
}

int ascii_0_byte_change() {
	char a[1000];
	int i;
	for (i = 0; i < 1000; i++) {
		a[i] = -1 - i;
	}
	printf("%d" ,strlen(a));
	return 0;
}

unsigned char i = 1;
int unsigned_byte_overflow() {
	for (; i <=255 ;i++) {
		puts("hello word\n");
	}
	return 0;
}