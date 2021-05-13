#include <stdio.h>
#define MAX 255

int compoments_codes() {
	char a = -1;
	signed char b = -1;
	unsigned char c = -1;
	printf("a=%d ,b=%d ,c=%d" ,a ,b ,c);
	return 0;
}

int main() {
	unsigned char a[MAX], i;
	for (i = 0; i < MAX ; i++) {
		a[i] = i;
		for (i = 0; i < MAX;i++) {
			printf("%d ",a[i]);
		}
	}
	return 0;
}