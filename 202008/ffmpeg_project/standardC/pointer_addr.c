#include <stdio.h>

int main() {
	int n = 10;
	char* pc = (char*)&n; //pointer to int lower byte address
	int* pi = &n;
	printf("%p\n" ,&n);
	printf("%p\n", pc);
	printf("%p\n", pc + 1);
	printf("%p\n", pi);
	printf("%p\n", pi + 1);

	return 0;
}