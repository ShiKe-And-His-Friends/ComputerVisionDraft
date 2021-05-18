#include <stdio.h>

int main() {
	int i = 0;
	if (i) {
		printf("int 0 select true\n");
	}
	else {
		printf("int 0 select false\n");
	}
	i = -1;
	if (i) {
		printf("int -1 select true\n");
	}
	else {
		printf("int -1 select false\n");
	}
	i = 100;
	if (i) {
		printf("int 100 select true\n");
	}
	else {
		printf("int 100 select false\n");
	}
	char a = 0;
	if (a) {
		printf("char 0 select true\n");
	}
	else {
		printf("char 0 select false\n");
	}
	a = '0';
	if (a) {
		printf("char 0 letter select true\n");
	}
	else {
		printf("char 0 letter select false\n");
	}
	int* p = NULL;
	if (p) {
		printf("int pointer select true\n");
	}
	else {
		printf("int pointer select false\n");
	}
}