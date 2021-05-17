#include <stdio.h>

int pointer_foot_long() {
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

int pointer_soft_value() {
	int n = 0x11223344;
	char* pc = (char*)&n;
	int *pi = &n;
	*pc = 0x55;
	printf("%p\n" ,pc);
	printf("%x\n", n);
	*pi = 0;
	printf("%p\n", pi);
	printf("%x\n", n);

	return 0;
}

int main() {
	int a = 10;
	int *pa = &a;
	int **ppa = &pa;
	printf("%p\n" ,&pa);
	printf("%x\n", ppa);

	**ppa = 30;
	printf("%x\n", a);
	printf("%x\n", ppa);

	return 0;
}